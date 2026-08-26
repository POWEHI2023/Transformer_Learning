# 当前全部 Attention 的 Mask 只支持了右侧 Padding
# TODO 端到端如何支持左侧 Padding ?

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from model.position.zposition import RoPE
from configs.zconfig import AttentionConfig, ParallelConfig

@dataclass
class AttentionOutput:
    output: torch.Tensor
    cached_key_value: tuple[torch.Tensor, torch.Tensor] | None = None

class BaseAttention(torch.nn.Module, ABC):
    def __init__(
        self,
        d_model: int,
        config: AttentionConfig,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        super().__init__()
        config.validate(d_model)

        assert d_model % config.n_heads == 0
        self.d_model = d_model
        self.n_heads = config.n_heads
        self.d_heads = self.d_model // self.n_heads

        # 针对当前唯一的 RoPE 位置编码
        assert self.d_heads % 2 == 0
        self.use_causal_mask = config.use_causal_mask

        _kind = config.kind

        if _kind == "gqa":
            self.head_per_group = self.n_heads // config.n_kv_heads
            self.n_kv_head = config.n_kv_heads

        if _kind == "mha":
            _iproj_odim = 3 * self.d_model
            self.n_kv_head = self.n_heads
        elif _kind == "mqa":
            _iproj_odim = self.d_model + 2 * self.d_heads
            self.n_kv_head = 1
        elif _kind == "gqa":
            _iproj_odim = self.d_model + 2 * (self.n_kv_head * self.d_heads)
            self.n_kv_head = config.n_kv_heads  # Group number
            # q head number for each kv head
            self.head_per_group = self.n_heads // self.n_kv_head
        else:
            raise ValueError(f"Unsupported Attention kind: {_kind}")

        self._iproj_odim = _iproj_odim
        self.input_proj = torch.nn.Linear(self.d_model, _iproj_odim)
        self.output_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.rope = RoPE(head_dim=self.d_heads, base=config.rope_base)

        self.attn_config = config
        self.parallel_config = parallel_config
        if self.parallel_config is not None:
            tp_size = self.parallel_config.tensor_parallel_size
            if self.n_heads % tp_size != 0:
                raise ValueError("")
            if self.n_kv_head % tp_size != 0:
                raise ValueError("")

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        # forbid past_key_value is not None and use_cache is False
        # because we need update cache in this case
        assert past_key_value is None or use_cache
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model
        # X -> QKV, 并且为 QK 添加位置编码
        q, k, v = self._preprocess_input(x, position_ids)
        (k, v), past_length = self._resolve_past_key_value(k, v, batch_size, past_key_value)
        blocked_mask = self._build_blocked_mask(q, k, past_length, attention_mask)

        _kind = self.attn_config.kind
        if _kind in ["mha", "mqa"]:
            # [B, H_q, S_q, D] @ [B, H_kv, D, S_kv] -> [B, H, S_q, S_kv]
            s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_heads)
            if blocked_mask is not None:
                s = s.masked_fill(blocked_mask, float("-inf"))
            # [B, H, S_q, S_k] @ [B, 1, S_kv, D_h] -> [B, H, S_q, D_h]
            o = torch.softmax(s, dim=-1) @ v
            # [B, S_q, D_m]
            o = o.transpose(1, 2).reshape(batch_size, sequence_length, d_model)

            return AttentionOutput(
                output=self.output_proj(o),
                cached_key_value=(k, v) if use_cache else None,
            )

        # Group Query Attention --- 当前还没有 MLA, 因此这里执行 GQA
        assert _kind == "gqa"
        raise RuntimeError("...")

    def _resolve_past_key_value(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        if past_key_value is not None:
            cached_k, cached_v = past_key_value
            assert cached_k.shape == cached_v.shape
            assert cached_k.ndim == 4 and cached_v.ndim == 4
            assert cached_k.shape[:2] == (batch_size, self.n_kv_head) and cached_k.shape[-1] == self.d_heads
            assert cached_v.shape[:2] == (batch_size, self.n_kv_head) and cached_v.shape[-1] == self.d_heads
            past_length = cached_k.shape[2]
            # [B, 1, S+T, d_head]
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)
        # k is RoPEd now, and we need causal mask and attention mask later
        else: past_length = 0

        return ((k, v), past_length)

    def _preprocess_input(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        qkv: torch.Tensor = self.input_proj(x)

        _split_shape = [self.d_model]
        _kind = self.attn_config.kind
        if _kind == "mha":
            _split_shape += [self.d_model] * 2
        elif _kind == "mqa":
            _split_shape += [self.d_heads] * 2
        elif _kind == "gqa":
            _split_shape += [self.d_heads * self.n_kv_head] * 2
        else: raise ValueError(f"Unsupported Attention kind: {_kind}")

        q, k, v = qkv.split(_split_shape, dim=-1)
        # to [B, H, S_c, D]
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_heads).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_kv_head, self.d_heads).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_kv_head, self.d_heads).transpose(1, 2)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        return (q, k, v)

    def _preprocess_input_local(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deprecated experimental prototype for parsing a local TP QKV layout.

        This method assumes a special cross-interleaved local QKV layout.  It is
        retained only as an experiment and must not be used by any Attention
        forward path.  Production TP should be provided by the parallel layer
        adapter instead.
        """
        warnings.warn(
            "_preprocess_input_local() is a deprecated experimental prototype "
            "and must not be used by Attention forward paths",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.parallel_config is None or self.parallel_config.tensor_parallel_size == 1:
            return self._preprocess_input(x, position_ids)

        batch_size, sequence_lengt, d_model = x.shape
        assert d_model == self.d_model

        # 总的 q heads 数量为 n_heads, 总的 kv heads 数量为 kv_heads.
        # [B, S, D] -> [B, S, QKV_D]
        qkv: torch.Tensor = self.input_proj(x)
        local_width = qkv.shape[-1]
        if self._iproj_odim % local_width != 0:
            raise ValueError("")
        _ratio = self._iproj_odim // local_width
        if self.parallel_config is None or self.parallel_config.tensor_parallel_size != _ratio:
            raise RuntimeError("")

        local_q_heads = self.n_heads // _ratio
        local_kv_heads = self.n_kv_head // _ratio
        # local_width 中包含 local_q_heads + 2 * local_kv_heads 个 head, [B, S, local_width]
        if local_width != (local_q_heads + 2 * local_kv_heads) * self.d_heads:
            raise ValueError("")
        if local_q_heads % local_kv_heads != 0:
            raise ValueError("")
        _cross_layout_ratio = local_q_heads // local_kv_heads
        _cross_layout_d = (_cross_layout_ratio + 2) * self.d_heads
        _cross_layout_n = local_width // _cross_layout_d
        qkv = qkv.view(batch_size, sequence_lengt, _cross_layout_n, _cross_layout_d)
        # [B, S, _cross_layout_n, _cross_layout_ratio]
        # [B, S, _cross_layout_n, d_head]
        # [B, S, _cross_layout_n, d_head]
        q, k, v = qkv.split(
            [_cross_layout_ratio * self.d_heads, self.d_heads, self.d_heads],
            dim=-1,
        )

        q = q.reshape(batch_size, sequence_lengt, local_q_heads, self.d_heads).transpose(1, 2)
        k = k.reshape(batch_size, sequence_lengt, local_kv_heads, self.d_heads).transpose(1, 2)
        v = v.reshape(batch_size, sequence_lengt, local_kv_heads, self.d_heads).transpose(1, 2)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        return (q, k, v)

    def _build_blocked_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        past_length: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        batch_size, _, sequence_length, _ = q.shape
        q_len, k_len = sequence_length, k.shape[2]
        query_positions = past_length + torch.arange(q_len, device=q.device)
        key_positions = torch.arange(k_len, device=q.device)

        # q @ k^T -> [q_len, k_len]
        # Query 位置 i 只能读取 Key 位置 <= i, 当 Key Position > Query Position 就需要被屏蔽
        # Key Position: [None, :] -> [[0, 1, 2, 3]], 表示每一列对应的 Key 的位置
        # Query Position: [:, None] -> [[0], [1], [2], [3]], 表示每一行对应 Query 的位置
        # 广播之后 mask 的形状为 [Q, K], 每个位置表示 Q Pos < K Pos 时为 True 需要屏蔽, 否则为 False
        blocked_mask = (
            key_positions[None, :] > query_positions[:, None]
            if self.use_causal_mask
            else None
        )

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, k_len)
            valid_keys = attention_mask.to(device=q.device, dtype=torch.bool)
            if not valid_keys.any(dim=-1).all():
                raise ValueError("Every sample must contain at least one valid key")
            valid_mask = ~valid_keys[:, None, None, :]
            blocked_mask = (
                valid_mask
                if blocked_mask is None
                else valid_mask | blocked_mask
            )
        return blocked_mask


class MultiHeadAttention(BaseAttention):
    def __init__(
        self,
        d_model: int,
        config: AttentionConfig,
    ) -> None:
        if config.kind != "mha" or config.backend != "eager":
            raise ValueError(
                "MultiHeadAttention: AttentionConfig kind should be mha:eager,"
                f"but got {config.kind}:{config.backend}"
            )
        super().__init__(d_model=d_model, config=config)


class MultiQueryAttention(BaseAttention):
    def __init__(
        self, 
        d_model: int, 
        config: AttentionConfig,
    ) -> None:
        if config.kind != "mqa" or config.backend != "eager":
            raise ValueError(
                "MultiQueryAttention: AttentionConfig kind should be mqa:eager,"
                f"but got {config.kind}:{config.backend}"
            )
        super().__init__(d_model, config)
    
        
class GroupQueryAttention(BaseAttention):
    def __init__(
        self,
        d_model: int,
        config: AttentionConfig,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        if config.kind != "gqa" or not config.backend in ["eager", "einsum"]:
            raise ValueError(
                "GroupQueryAttention: AttentionConfig kind should be gqa:,"
                f"but got {config.kind}:{config.backend}"
            )
        super().__init__(d_model, config, parallel_config)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        q, k, v = self._preprocess_input(x, position_ids)
        (k, v), past_length = self._resolve_past_key_value(k, v, batch_size, past_key_value)
        blocked_mask = self._build_blocked_mask(q, k, past_length, attention_mask)

        if self.attn_config.backend == "eager":
            local_outputs = []
            for group_index in range(self.n_kv_head):
                q_head_start = group_index * self.head_per_group
                q_head_end = q_head_start + self.head_per_group

                # [B, HeadPerGroup, S, D]
                q_local: torch.Tensor = q[:, q_head_start : q_head_end]
                # [B, 1, HS + CS, D]
                k_local: torch.Tensor = k[:, group_index : group_index + 1]
                v_local: torch.Tensor = v[:, group_index : group_index + 1]

                # [B, HeadPerGroup, S, D] @ [B, 1, D, HS + CS] -> [B, HeadPerGroup, S, HS + CS]
                s_local = q_local @ k_local.transpose(-2, -1) / math.sqrt(self.d_heads)

                if blocked_mask is not None:
                    s_local = s_local.masked_fill(blocked_mask, float("-inf"))

                # [B, HeadPerGroup, S, HS + CS] @ [B, 1, HS + CS, D] -> [B, HeadPerGroup, S, D]
                attention_weights = torch.softmax(s_local, dim=-1)
                o_local = attention_weights @ v_local
                local_outputs.append(o_local)

            # [B, n_head, S, d_head]
            o = torch.cat(local_outputs, dim=1)
            o = o.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)

            return AttentionOutput(
                output=self.output_proj(o),
                cached_key_value=(k, v) if use_cache else None,
            )

        # Group Query Attention --- 支持另一个 backed 为 einsum
        assert self.attn_config.backend == "einsum"

        # [B, G, R, Q, D]
        q_grouped = q.reshape(batch_size, self.n_kv_head, self.head_per_group, sequence_length, self.d_heads)
        # [B, G, R, Q, K]
        s = torch.einsum(
            "bgrqd,bgkd->bgrqk",
            q_grouped, k,
        ) / math.sqrt(self.d_heads)

        if blocked_mask is not None:
            # [Q, K] -> [1, Q, K] -> [1, Q, K]
            # [B, 1, Q, K] -> [B, 1, 1, Q, K]
            s = s.masked_fill(blocked_mask.unsqueeze(-3), float("-inf"))

        attention_weights = torch.softmax(s, dim=-1)
        # [B, G, R, Q, D]
        output_grouped = torch.einsum(
            "bgrqk,bgkd->bgrqd",
            attention_weights, v,
        )

        o = output_grouped.view(batch_size, self.n_heads, sequence_length, self.d_heads)
        o = o.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)

        return AttentionOutput(
            output=self.output_proj(o),
            cached_key_value=(k, v) if use_cache else None,
        )


def build_attention(
    d_model: int,
    config: AttentionConfig
) -> MultiHeadAttention | MultiQueryAttention | GroupQueryAttention:
    if config.kind == "mha": _module = MultiHeadAttention
    elif config.kind == "mqa": _module = MultiQueryAttention
    elif config.kind == "gqa": _module = GroupQueryAttention
    else: raise ValueError(f"Unsupported Attention kind: {config.kind}")

    return _module(d_model, config)
