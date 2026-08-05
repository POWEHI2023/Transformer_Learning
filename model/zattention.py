from __future__ import annotations

import math
import torch
from model.zposition import RoPE

class Attention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int = 512,
        use_causal_mask: bool = True,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.n_heads
        assert self.head_dim % 2 == 0
        self.use_causal_mask = use_causal_mask

        self.input_proj = torch.nn.Linear(self.d_model, 3 * self.d_model)
        self.output_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.rope = RoPE(head_dim=self.head_dim, base=rope_base)
    
    # X  shape: [B, L, d_model]
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.shape
        # [B, L, 3 * d_model]
        x = self.input_proj(x)
        # split the last dimention into 3 partitions, [B, L, d_model]
        q, k, v = x.chunk(3, dim=-1)

        # [B, n_heads, L, head_dim]
        q = q.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE: position embedding
        # position_ids = torch.arange(sequence_length, device=x.device)
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # PREFILL: cached kv
        k_cache, v_cache = k, v

        # Standard Attention for each Head
        # S = QK / sqrt(n)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        blocked_mask = None

        if self.use_causal_mask:
            q_len = q.size(-2)
            k_len = k.size(-2)
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            # s = s.masked_fill(causal_mask, float("-inf"))
            blocked_mask = causal_mask

        # Use attention mask
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, sequence_length)
            # [B, L] -> [B, 1, 1, L]
            padding_blocked = ~attention_mask.to(device=q.device, dtype=torch.bool)[:, None, None, :]
            blocked_mask = (
                padding_blocked
                if blocked_mask is None
                else blocked_mask | padding_blocked
            )

        if blocked_mask is not None:
            s = s.masked_fill(blocked_mask, float("-inf"))

        # O = (QK / sqrt(n)) @ V
        o = torch.softmax(s, dim=-1) @ v
        # combine all heads
        o = o.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)

        return self.output_proj(o), k_cache, v_cache

# TODO fuse MHA into one
class DecodeAttention_MHA(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int = 512,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = self.d_model // self.n_heads
        assert self.d_head % 2 == 0

        self.input_proj = torch.nn.Linear(self.d_model, 3 * self.d_model)
        self.output_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.rope = RoPE(head_dim=self.d_head, base=rope_base)

    def forward(
        self, 
        x: torch.Tensor,        # [B, 1, d_model]
        cached_k: torch.Tensor, # [B, H, S, d_model], RoPEd K
        cached_v: torch.Tensor, # [B, H, S, d_model]
        position_ids: torch.Tensor,                 # [1] or [B, 1]
        attention_mask: torch.Tensor | None = None, # [B, S + 1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length, d_model = x.shape
        assert query_length == 1, "DecodeAttention_MHA only supports single-token decode"
        assert d_model == self.d_model
        
        x = self.input_proj(x) # [B, 1, 3 * d_model]
        q, k, v = x.chunk(3, dim=-1)

        # [B, n_head, 1, d_head]
        q = q.view(batch_size, 1, self.n_heads, self.d_head).transpose(2, 1)
        # [B, n_head, S + 1, d_head]
        k = k.view(batch_size, 1, self.n_heads, self.d_head).transpose(2, 1)
        v = v.view(batch_size, 1, self.n_heads, self.d_head).transpose(2, 1)

        # apply rope
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # cached kv shape: [B, n_head, S, d_head]
        assert cached_k.shape[:2] == (batch_size, self.n_heads) and cached_k.shape[-1] == self.d_head
        assert cached_v.shape[:2] == (batch_size, self.n_heads) and cached_v.shape[-1] == self.d_head
        assert cached_k.shape[-2] == cached_v.shape[-2] # S of K == S of V

        # 2 * [B, n_head, S + 1, d_head]
        # this is not efficial, re-allocate in k_cache then straight kv_cache[:, :, cache_position] = kv
        k_cache = torch.cat([cached_k, k], dim=2)
        v_cache = torch.cat([cached_v, v], dim=2)
        cached_length = k_cache.shape[2]

        # [B, H, 1, D] @ [B, H, D, S+1] -> [B, H, 1, S+1]
        s: torch.Tensor = (q @ k_cache.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, cached_length)
            valid_keys = attention_mask.to(device=s.device, dtype=torch.bool)

            assert valid_keys.any(dim=-1).all(), "Every sample must have at least one valid cached token"
            # [B, S+1] -> [B, 1, 1, S+1]
            blocked_key = ~valid_keys[:, None, None, :]
            s = s.masked_fill(blocked_key, float("-inf"))

        # Single token decode do not need causal mask, there is not token of future in cache

        attention_weights = torch.softmax(s, dim=-1)
        # [B, H, 1, S+1] @ [B, H, S+1, D] -> [B, H, 1, D]
        o = attention_weights @ v_cache
        # [B, 1, D]
        o = o.transpose(2, 1).reshape(batch_size, 1, self.d_model)
        o = self.output_proj(o) # [B, 1, D]

        return o, k_cache, v_cache


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
        # 生成 causal mask 支持 Decoder-only Transformer
        # 不生成 causal mask 支持双向 Attention 或 Cross-Attention
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
        # q @ k^T -> [q_len, k_len]
        # Query 位置 i 只能读取 Key 位置 <= i, 当 Key Position > Query Position 就需要被屏蔽
        # Key Position: [None, :] -> [[0, 1, 2, 3]], 表示每一列对应的 Key 的位置
        # Query Position: [:, None] -> [[0], [1], [2], [3]], 表示每一行对应 Query 的位置
        # 广播之后 mask 的形状为 [Q, K], 每个位置表示 Q Pos < K Pos 时为 True 需要屏蔽, 否则为 False
        if self.use_causal_mask:
            causal_mask = key_positions[None, :] > query_positions[:, None] # 可以满足 Chunked Decode
            # [Q, K]
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
    pass

class InferAttention_MLA(torch.nn.Module):
    pass