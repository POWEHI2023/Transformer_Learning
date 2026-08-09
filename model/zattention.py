# 当前全部 Attention 的 Mask 只支持了右侧 Padding
# TODO 端到端如何支持左侧 Padding ?

from __future__ import annotations

import math
import torch
from abc import ABC, abstractmethod
from model.zposition import RoPE
from configs.zconfig import AttentionConfig
from dataclasses import dataclass

@dataclass
class AttentionOutput:
    output: torch.Tensor
    cached_key_value: tuple[torch.Tensor, torch.Tensor] | None = None

class BaseAttention(torch.nn.Module, ABC):
    def __init__(
        self,
        d_model: int,
        config: AttentionConfig,
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
            self.n_kv_head = config.n_kv_heads
        else:
            raise ValueError(f"Unsupported Attention kind: {_kind}")
        
        self.input_proj = torch.nn.Linear(self.d_model, _iproj_odim)
        self.output_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.rope = RoPE(head_dim=self.d_heads, base=config.rope_base)

        self.attn_config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        raise NotImplementedError

    def _resolve_past_key_value(
        self,
        batch_size: int,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        cached_k, cached_v = past_key_value
        assert cached_k.shape == cached_v.shape
        assert cached_k.ndim == 4 and cached_v.ndim == 4
        assert cached_k.shape[:2] == (batch_size, self.n_kv_head) and cached_k.shape[-1] == self.d_heads
        assert cached_v.shape[:2] == (batch_size, self.n_kv_head) and cached_v.shape[-1] == self.d_heads
        past_length = cached_k.shape[2]
        return ((cached_k, cached_v), past_length)

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


# Pure Multi-Head Attention
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

        if past_key_value is not None:
            (cached_k, cached_v), past_length = \
                self._resolve_past_key_value(batch_size, past_key_value)
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)
        else: past_length = 0
        blocked_mask = self._build_blocked_mask(q, k, past_length, attention_mask)

        # S = QK / sqrt(n)
        s: torch.Tensor = q @ k.transpose(-2, -1) / math.sqrt(self.d_heads)
        if blocked_mask is not None:
            s = s.masked_fill(blocked_mask, float("-inf"))
        # O = (QK / sqrt(n)) @ V
        o = torch.softmax(s, dim=-1) @ v
        # Combine all heads
        o = o.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)
        return AttentionOutput(
            output=self.output_proj(o),
            cached_key_value=(k, v) if use_cache else None,
        )


class MultiQueryAttention(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        use_causal_mask: bool = True, 
        rope_base: float = 10_000.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0
        self.d_head = self.d_model // self.n_head
        assert self.d_head % 2 == 0
        self.use_causal_mask = use_causal_mask

        self.input_proj = torch.nn.Linear(self.d_model, self.d_model + 2 * self.d_head)
        self.output_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.rope = RoPE(head_dim=self.d_head, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,    # [B, T, d_model]
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,    # default do training
    ) -> tuple[
        torch.Tensor, 
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        # forbid past_key_value is not None and use_cache is False
        # because we need update cache in this case
        assert past_key_value is None or use_cache

        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        # Q: [B, H, T, d_head], KV: [B, 1, T, d_head]
        qkv: torch.Tensor = self.input_proj(x) # [B, T, d_model + 2 * d_head]
        q, k, v = qkv.split([self.d_model, self.d_head, self.d_head], dim=-1)

        q = q.view(batch_size, sequence_length, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, sequence_length, 1, self.d_head).transpose(1, 2)
        v = v.view(batch_size, sequence_length, 1, self.d_head).transpose(1, 2)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        if past_key_value is not None:
            # [B, 1, S, d_head]
            k_cached, v_cached = past_key_value
            assert k_cached.ndim == 4 and v_cached.ndim == 4
            assert k_cached.shape == v_cached.shape
            assert k_cached.shape[0] == batch_size
            assert k_cached.shape[1] == 1
            assert k_cached.shape[-1] == self.d_head

            past_length = k_cached.size(-2)

            # Decoder: [B, 1, S+T, d_head]
            k = torch.cat([k_cached, k], dim=-2)
            v = torch.cat([v_cached, v], dim=-2)
        else: past_length = 0

        # [B, H, T, D] @ [B, 1, D, S] -> [B, H, T, S]
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        q_len = q.shape[2]
        k_len = k.shape[2]

        blocked_mask = None
        query_positions = past_length + torch.arange(q_len, device=q.device)
        key_positions = torch.arange(k_len, device=q.device)

        if self.use_causal_mask:
            causal_mask = key_positions[None, :] > query_positions[:, None]
            blocked_mask = causal_mask

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, k_len)
            # [B, 1, 1, K]
            attention_mask = ~attention_mask.to(device=s.device, dtype=torch.bool)[:, None, None, :]
            # [B, 1, Q, K]
            blocked_mask = (
                attention_mask
                if blocked_mask is None
                else blocked_mask | attention_mask
            )

        if blocked_mask is not None:
            s = s.masked_fill(blocked_mask, float("-inf"))

        o = torch.softmax(s, dim=-1) @ v
        o = o.transpose(1, 2).reshape(batch_size, sequence_length, d_model)

        return self.output_proj(o), (k, v) if use_cache else None

        
class GroupQueryAttention(torch.nn.Module):
    '''
    MHA: Query 的每个 Head 使用一个 KV Head, 代表 Query 的 Head 个数和 KV 是一样的
    MQA: Query 的所有 Head 使用一个 KV Head, 代表一个 Query 只有一个 KV Head
    GQA: Query 中的一组 Head 使用一个 KV Head, 代表 KV 的 Head 个数为 Query Head // Group
    '''
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        use_causal_mask: bool = True,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        assert n_kv_head > 0 and n_head > 0 and d_model > 0
        assert d_model % n_head == 0
        assert n_head % n_kv_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = self.d_model // self.n_head
        self.n_kv_head = n_kv_head
        self.head_per_group = self.n_head // self.n_kv_head

        self.use_causal_mask = use_causal_mask

        self.kv_d_model = self.n_kv_head * self.d_head

        self.input_proj = torch.nn.Linear(self.d_model, self.d_model + 2 * self.kv_d_model)
        self.output_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.rope = RoPE(head_dim=self.d_head, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,    # [B, S, d_model]
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        qkv: torch.Tensor = self.input_proj(x) # [B, S, d_model + kv_d_model + kv_d_model]
        q, k, v = qkv.split(
            [self.d_model, self.kv_d_model, self.kv_d_model],
            dim=-1,
        )

        # [B, H, S, D]
        q = q.view(batch_size, sequence_length, self.n_head, self.d_head).transpose(1, 2)
        # [B, G, S, D]
        k = k.view(batch_size, sequence_length, self.n_kv_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_kv_head, self.d_head).transpose(1, 2)

        # append position for each position
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        if past_key_value is not None:
            # in this case, x can be chunked request, so the S uppon means chunked sequence length
            # make chunked sequence length as `ChunkS`

            # [B, G, HistoryS, D]
            k_cached, v_cached = past_key_value
            assert k_cached.ndim == 4 and v_cached.ndim == 4
            assert k_cached.shape == v_cached.shape
            assert k_cached.shape[:2] == (batch_size, self.n_kv_head)
            assert k_cached.shape[-1] == self.d_head

            # History Sequence Length
            past_length = k_cached.shape[2]

            # [B, G, HistoryS + ChunkS, D]
            k = torch.cat([k_cached, k], dim=-2)
            v = torch.cat([v_cached, v], dim=-2)
        # k is RoPEd now, and we need causal mask and attention mask later
        else: past_length = 0

        q_len, k_len = q.shape[2], k.shape[2]
        query_positions = past_length + torch.arange(q_len, device=q.device)
        key_positions = torch.arange(k_len, device=q.device)

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

            # [B, 1, 1, K_len]
            attention_mask = ~attention_mask.to(device=q.device, dtype=torch.bool)[:, None, None, :]
            blocked_mask = (
                attention_mask  # [B, 1, 1, K_len]
                if blocked_mask is None
                else blocked_mask | attention_mask  # [B, 1, Q, K_len]
            )

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
            s_local = q_local @ k_local.transpose(-2, -1) / math.sqrt(self.d_head)

            if blocked_mask is not None:
                s_local = s_local.masked_fill(blocked_mask, float("-inf"))

            # [B, HeadPerGroup, S, HS + CS] @ [B, 1, HS + CS, D] -> [B, HeadPerGroup, S, D]
            attention_weights = torch.softmax(s_local, dim=-1)
            o_local = attention_weights @ v_local
            local_outputs.append(o_local)

        # [B, n_head, S, d_head]
        o = torch.cat(local_outputs, dim=1)
        o = o.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)

        return self.output_proj(o), (k, v) if use_cache else None

class GroupQueryAttention_V2(GroupQueryAttention):
    '''
    einsum 是一种用下标描述张量乘法、广播和归约的接口，让 GQA 的数学结构很清楚。
    '''
    def forward(
        self,
        x: torch.Tensor,    # [B, S, d_model]
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        qkv: torch.Tensor = self.input_proj(x) # [B, S, d_model + kv_d_model + kv_d_model]
        q, k, v = qkv.split(
            [self.d_model, self.kv_d_model, self.kv_d_model],
            dim=-1,
        )

        # [B, H, S, D]
        q = q.view(batch_size, sequence_length, self.n_head, self.d_head).transpose(1, 2)
        # [B, G, S, D]
        k = k.view(batch_size, sequence_length, self.n_kv_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_kv_head, self.d_head).transpose(1, 2)

        # append position for each position
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        if past_key_value is not None:
            # in this case, x can be chunked request, so the S uppon means chunked sequence length
            # make chunked sequence length as `ChunkS`

            # [B, G, HistoryS, D]
            k_cached, v_cached = past_key_value
            assert k_cached.ndim == 4 and v_cached.ndim == 4
            assert k_cached.shape == v_cached.shape
            assert k_cached.shape[:2] == (batch_size, self.n_kv_head)
            assert k_cached.shape[-1] == self.d_head

            # History Sequence Length
            past_length = k_cached.shape[2]

            # [B, G, HistoryS + ChunkS, D]
            k = torch.cat([k_cached, k], dim=-2)
            v = torch.cat([v_cached, v], dim=-2)
        # k is RoPEd now, and we need causal mask and attention mask later
        else: past_length = 0

        q_len, k_len = q.shape[2], k.shape[2]
        query_positions = past_length + torch.arange(q_len, device=q.device)
        key_positions = torch.arange(k_len, device=q.device)

        blocked_mask = None
        if self.use_causal_mask:
            # [Q, K] -> [1, 1, 1, Q, K]
            blocked_mask = (key_positions[None, :] > query_positions[:, None])[None, None, None, :, :]

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, k_len)
            valid_keys = attention_mask.to(device=q.device, dtype=torch.bool)
            if not valid_keys.any(dim=-1).all():
                raise ValueError("Every sample must contain at least one valid key")

            # [B, 1, 1, 1, K_len]
            attention_mask = ~valid_keys[:, None, None, None, :]
            # [B, 1, 1, Q, K_len]
            blocked_mask = (
                attention_mask
                if blocked_mask is None
                else blocked_mask | attention_mask
            )

        # [B, G, R, Q, D]
        q_grouped = q.reshape(batch_size, self.n_kv_head, self.head_per_group, sequence_length, self.d_head)
        # [B, G, R, Q, K]
        s = torch.einsum(
            "bgrqd,bgkd->bgrqk",
            q_grouped, k,
        ) / math.sqrt(self.d_head)

        if blocked_mask is not None:
            s = s.masked_fill(blocked_mask, float("-inf"))

        attention_weights = torch.softmax(s, dim=-1)
        # [B, G, R, Q, D]
        output_grouped = torch.einsum(
            "bgrqk,bgkd->bgrqd",
            attention_weights, v,
        )

        output = output_grouped.view(batch_size, self.n_head, sequence_length, self.d_head)
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)

        return self.output_proj(output), (k, v) if use_cache else None

class InferAttention_MLA(torch.nn.Module):
    pass