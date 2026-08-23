from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass

import torch

from configs.zconfig import AttentionConfig
from model.position.zposition import RoPE


@dataclass
class PagedAttentionOutput:
    """Output of a paged-attention invocation.

    ``seq_lens`` contains the sequence lengths after the new K/V entries have
    been appended.  The cache itself is updated in place so that its physical
    storage can be shared by all requests managed by a continuous-batching
    scheduler.
    """

    output: torch.Tensor
    seq_lens: torch.Tensor

'''
原先的 Shape: [B, Hkv, Q, D], 根据 Batch 构造缓存.
Block 中是以 Token 为单位的: [N, P, Hkv, D] -> N 个 Block, 每个 Block 中最多 P 个 token,
每个 token 被分成 Hkv 和 Head, 每个 Head 的维度是 D.
例如: [100, 16, 2, 64], 一共 100 个物理块, 每个块保存 16 个 token, 使用 2 和 KV Head, 每个 Head 维度是 64.
K 和 V 使用两块独立的存储:
physical block 0
├── token offset 0: [Hkv, D]
├── token offset 1: [Hkv, D]
├── ...
└── token offset 15: [Hkv, D]

physical block 1
├── token offset 0: [Hkv, D]
└── ...

对于单个请求, 看到的是连续的逻辑 token, 0 1 2 3 4 5 6 7 8 9 ...
可以为这些 token 划分连续的逻辑 Block, 但映射到物理上不一定连续, block_tables 就是逻辑到物理的映射.
block_tables: [B, M], 行代表独立的请求, 列代表逻辑块到物理块的映射, -1 表示尚未分配物理块, 调换请求顺序时只需要调换其中的行.

seq_lens: [B], 表示执行本次 Attention 前每个请求缓存了多少个 token.
slot_mapping: [B, Q], 本次的每个请求中, 每个新 token 应该写入哪个物理 cache slot. slot = phy_block_id * block_size + block_off.


-------------------------------
PyTorch reference 通过 block_tables 将分页 KV 临时收集成 [B, Hkv, Kmax, D], valid_keys: [B, Kmax]
然后把 Q 组织成统一的 Grouped Layout: [B, Hkv, R, Q, D], R = Hq // Hkv,
接下来计算 einsum "bgrqd,bgkd->bgrqk", Q_goruped, K -> [B, Hkv, R, Q, Kmax],
然后和 V 相乘: [B, Hkv, R, Q, Kmax] x [B, Hkv, Kmax, D] -> [B, Hkv, R, Q, D],
合并 KV Group 和组内 Q head -> [B, Hq, Q, D], 最后合并 head -> [B, Q, d_model]

padding_mask: valid_keys[b, k] = k < seq_lens[b] -> [B, Kmax]
causal mask 根据绝对位置计算: 
    query_position = past_seq_len + query_offset
    blocked = key_position > query_position
broadcast: [B, 1, 1, Q, Kmax]

key_positions[None, None, None, None, :] # [1, 1, 1, 1, Kmax]
query_positions[:, None, None, :, None] # [B, 1, 1, Q, 1]
---
真正高性能的 Triton/CUDA Paged Attention 不会执行完整 gather, 而是在 kernel 内直接计算避免 [B, Hkv, Kmax, D] 临时内存.
'''
@dataclass
class PagedKVCache:
    """Preallocated physical storage used by paged attention.

    Cache layout:

    - key_cache:   [num_blocks, block_size, n_kv_heads, head_dim]
    - value_cache: [num_blocks, block_size, n_kv_heads, head_dim]

    This class owns only the physical K/V storage.  Allocation and reclamation
    of block ids belong to the continuous-batching scheduler.  The scheduler
    passes the current requests' logical-to-physical ``block_tables`` to the
    attention layer on every invocation.
    """

    key_cache: torch.Tensor
    value_cache: torch.Tensor

    def __post_init__(self) -> None:
        if self.key_cache.shape != self.value_cache.shape:
            raise ValueError("key_cache and value_cache must have identical shapes")
        if self.key_cache.ndim != 4:
            raise ValueError(
                "paged K/V cache must have shape "
                "[num_blocks, block_size, n_kv_heads, head_dim]"
            )
        if self.key_cache.device != self.value_cache.device:
            raise ValueError("key_cache and value_cache must be on the same device")
        if self.key_cache.dtype != self.value_cache.dtype:
            raise ValueError("key_cache and value_cache must have the same dtype")
        if any(size <= 0 for size in self.key_cache.shape):
            raise ValueError("all paged K/V cache dimensions must be positive")

    @classmethod
    def empty(
        cls,
        *,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PagedKVCache:
        if min(num_blocks, block_size, n_kv_heads, head_dim) <= 0:
            raise ValueError("all paged K/V cache dimensions must be positive")

        shape = (num_blocks, block_size, n_kv_heads, head_dim)
        return cls(
            key_cache=torch.empty(shape, device=device, dtype=dtype),
            value_cache=torch.empty(shape, device=device, dtype=dtype),
        )

    @property
    def num_blocks(self) -> int:
        return self.key_cache.shape[0]

    @property
    def block_size(self) -> int:
        return self.key_cache.shape[1]

    @property
    def n_kv_heads(self) -> int:
        return self.key_cache.shape[2]

    @property
    def head_dim(self) -> int:
        return self.key_cache.shape[3]

    @property
    def device(self) -> torch.device:
        return self.key_cache.device

    @property
    def dtype(self) -> torch.dtype:
        return self.key_cache.dtype

    def _validate_block_tables(self, block_tables: torch.Tensor, batch_size: int) -> None:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            raise ValueError("block_tables must have shape [batch_size, max_blocks_per_request]")
        if block_tables.shape[1] <= 0:
            raise ValueError("block_tables must contain at least one logical block column")
        if block_tables.device != self.device:
            raise ValueError("block_tables and the K/V cache must be on the same device")
        if block_tables.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_tables must use torch.int32 or torch.int64")

    def build_slot_mapping(
        self,
        block_tables: torch.Tensor,
        start_positions: torch.Tensor,
        query_length: int,
        query_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Map each new logical token position to a flat physical cache slot.

        Invalid/padded query positions receive ``-1``.  A scheduler that has
        already calculated its slot mapping can pass that mapping directly to
        ``BasePagedAttention.forward`` and skip this helper.
        """

        batch_size = start_positions.numel()
        self._validate_block_tables(block_tables, batch_size)
        if query_length <= 0:
            raise ValueError("query_length must be positive")
        if start_positions.shape != (batch_size,) or query_lens.shape != (batch_size,):
            raise ValueError("start_positions and query_lens must have shape [batch_size]")

        offsets = torch.arange(query_length, device=self.device, dtype=torch.long)
        logical_positions = start_positions.to(torch.long)[:, None] + offsets[None, :]
        valid_queries = offsets[None, :] < query_lens.to(torch.long)[:, None]
        logical_blocks = torch.div(logical_positions, self.block_size, rounding_mode="floor")
        block_offsets = logical_positions.remainder(self.block_size)

        if valid_queries.any():
            max_logical_block = int(logical_blocks[valid_queries].max().item())
            if max_logical_block >= block_tables.shape[1]:
                raise ValueError("block_tables does not contain enough logical blocks")

        safe_logical_blocks = logical_blocks.masked_fill(~valid_queries, 0)
        physical_blocks = block_tables.to(torch.long).gather(1, safe_logical_blocks)
        valid_physical_blocks = physical_blocks[valid_queries]
        if valid_physical_blocks.numel() > 0:
            if (valid_physical_blocks < 0).any() or (
                valid_physical_blocks >= self.num_blocks
            ).any():
                raise ValueError("block_tables contains an invalid physical block id")

        slots = physical_blocks * self.block_size + block_offsets
        return slots.masked_fill(~valid_queries, -1)

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write projected K/V for valid new tokens into physical cache slots."""

        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("key and value must have shape [B, n_kv_heads, Q, head_dim]")

        batch_size, n_kv_heads, query_length, head_dim = key.shape
        if (n_kv_heads, head_dim) != (self.n_kv_heads, self.head_dim):
            raise ValueError("projected K/V shape does not match the paged cache")
        if key.device != self.device or value.device != self.device:
            raise ValueError("projected K/V and the paged cache must be on the same device")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("projected K/V and the paged cache must have the same dtype")
        if slot_mapping.shape != (batch_size, query_length):
            raise ValueError("slot_mapping must have shape [B, Q]")
        if slot_mapping.device != self.device:
            raise ValueError("slot_mapping and the paged cache must be on the same device")
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            raise TypeError("slot_mapping must use torch.int32 or torch.int64")

        valid = slot_mapping >= 0
        slots = slot_mapping[valid].to(torch.long)
        if slots.numel() == 0:
            return
        if (slots >= self.num_blocks * self.block_size).any():
            raise ValueError("slot_mapping contains a cache slot outside physical storage")
        if torch.unique(slots).numel() != slots.numel():
            raise ValueError("slot_mapping cannot write multiple tokens to the same cache slot")

        physical_blocks = torch.div(slots, self.block_size, rounding_mode="floor")
        block_offsets = slots.remainder(self.block_size)
        # [B, H_kv, Q, D] -> [B, Q, H_kv, D]
        key_by_token = key.transpose(1, 2)[valid]
        value_by_token = value.transpose(1, 2)[valid]

        # A PagedKVCache is inference state, not part of an autograd graph.
        with torch.no_grad():
            # Advanced indexing returns a temporary tensor, therefore copy_ on
            # that result would not update the cache.  index_put_ writes into
            # the original physical storage.
            self.key_cache.index_put_(
                (physical_blocks, block_offsets), key_by_token.detach()
            )
            self.value_cache.index_put_(
                (physical_blocks, block_offsets), value_by_token.detach()
            )

    def gather(
        self,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize each request's logical K/V sequence.

        This is a correctness-first PyTorch reference implementation.  It
        returns dense tensors padded to ``max(seq_lens)``.  An optimized Triton
        or CUDA paged-attention kernel should read blocks through block_tables
        directly and avoid this materialization.
        """

        batch_size = seq_lens.numel()
        self._validate_block_tables(block_tables, batch_size)
        if seq_lens.shape != (batch_size,):
            raise ValueError("seq_lens must have shape [batch_size]")
        if seq_lens.device != self.device:
            raise ValueError("seq_lens and the paged cache must be on the same device")
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("seq_lens must use torch.int32 or torch.int64")
        if (seq_lens < 0).any():
            raise ValueError("seq_lens cannot contain negative lengths")

        max_sequence_length = int(seq_lens.max().item()) if batch_size > 0 else 0
        if max_sequence_length == 0:
            shape = (batch_size, self.n_kv_heads, 0, self.head_dim)
            empty = self.key_cache.new_empty(shape)
            return empty, empty.clone(), torch.zeros(
                (batch_size, 0), device=self.device, dtype=torch.bool
            )

        positions = torch.arange(max_sequence_length, device=self.device, dtype=torch.long)
        valid_keys = positions[None, :] < seq_lens.to(torch.long)[:, None]
        logical_blocks = positions // self.block_size
        if int(logical_blocks[-1].item()) >= block_tables.shape[1]:
            raise ValueError("block_tables does not contain enough logical blocks")

        logical_blocks = logical_blocks[None, :].expand(batch_size, -1)
        physical_blocks = block_tables.to(torch.long).gather(1, logical_blocks)
        valid_physical_blocks = physical_blocks[valid_keys]
        if (valid_physical_blocks < 0).any() or (
            valid_physical_blocks >= self.num_blocks
        ).any():
            raise ValueError("block_tables contains an invalid physical block id")

        safe_physical_blocks = physical_blocks.masked_fill(~valid_keys, 0)
        block_offsets = positions.remainder(self.block_size)[None, :].expand(batch_size, -1)
        # Advanced indexing produces [B, K, H_kv, D].
        key = self.key_cache[safe_physical_blocks, block_offsets]
        value = self.value_cache[safe_physical_blocks, block_offsets]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        return key, value, valid_keys


class BasePagedAttention(torch.nn.Module, ABC):
    """Shared paged-attention implementation for MHA, MQA and GQA.

    The implementation supports continuous batching because physical cache
    storage has no active-batch dimension.  ``block_tables`` and ``seq_lens``
    describe only the requests participating in the current model invocation.

    The current implementation is an eager PyTorch reference.  It demonstrates
    cache allocation contracts and numerical behavior, but it is not yet a
    fused high-performance paged-attention kernel.
    """

    def __init__(
        self,
        d_model: int,
        config: AttentionConfig,
        *,
        n_kv_heads: int,
    ) -> None:
        super().__init__()
        config.validate(d_model)
        if n_kv_heads <= 0 or config.n_heads % n_kv_heads != 0:
            raise ValueError("n_kv_heads must be positive and divide n_heads")

        self.d_model = d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // config.n_heads
        self.heads_per_kv_head = config.n_heads // n_kv_heads
        self.use_causal_mask = config.use_causal_mask
        self.attn_config = config

        projection_width = (
            self.n_heads + 2 * self.n_kv_heads
        ) * self.head_dim
        self.input_proj = torch.nn.Linear(d_model, projection_width)
        self.output_proj = torch.nn.Linear(d_model, d_model)
        self.rope = RoPE(head_dim=self.head_dim, base=config.rope_base)

    def create_kv_cache(
        self,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PagedKVCache:
        parameter = self.input_proj.weight
        return PagedKVCache.empty(
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length, _ = x.shape
        qkv = self.input_proj(x)
        q_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        q, key, value = qkv.split((q_width, kv_width, kv_width), dim=-1)

        q = q.view(batch_size, query_length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(
            batch_size, query_length, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, query_length, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        return self.rope(q, position_ids), self.rope(key, position_ids), value

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, blocked_mask: torch.Tensor) -> torch.Tensor:
        """Softmax that returns zeros, rather than NaNs, for fully masked rows."""

        scores_float = scores.float().masked_fill(blocked_mask, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores_float, dim=-1)
        weights = weights.masked_fill(blocked_mask, 0.0)
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(denominator > 0, weights / denominator.clamp_min(1e-20), weights)
        return weights.to(scores.dtype)

    def _compute_attention(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_keys: torch.Tensor,
        past_seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, query_length, _ = q.shape
        key_length = key.shape[2]

        # Treat every KV head and its associated Q heads as one group.  For
        # MHA R=1, for MQA G=1, and for GQA both G and R can be greater than 1.
        q_grouped = q.reshape(
            batch_size,
            self.n_kv_heads,
            self.heads_per_kv_head,
            query_length,
            self.head_dim,
        )
        scores = torch.einsum("bgrqd,bgkd->bgrqk", q_grouped, key)
        scores = scores / math.sqrt(self.head_dim)

        key_positions = torch.arange(key_length, device=q.device, dtype=torch.long)
        query_offsets = torch.arange(query_length, device=q.device, dtype=torch.long)
        query_positions = past_seq_lens.to(torch.long)[:, None] + query_offsets[None, :]
        valid_queries = query_offsets[None, :] < query_lens.to(torch.long)[:, None]

        blocked = ~valid_keys[:, None, None, None, :]
        if self.use_causal_mask:
            causal = key_positions[None, None, None, None, :] > query_positions[
                :, None, None, :, None
            ]
            blocked = blocked | causal
        blocked = blocked | ~valid_queries[:, None, None, :, None]

        weights = self._masked_softmax(scores, blocked)
        output = torch.einsum("bgrqk,bgkd->bgrqd", weights, value)
        output = output.reshape(batch_size, self.n_heads, query_length, self.head_dim)
        output = output.transpose(1, 2).reshape(batch_size, query_length, self.d_model)
        return output, valid_queries

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: PagedKVCache,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        query_lens: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> PagedAttentionOutput:
        """Append K/V to the paged cache and attend over each request.

        Args:
            x: New hidden states with shape [B, Q, d_model].
            kv_cache: Global physical cache shared by all active requests.
            block_tables: [B, max_blocks_per_request].  Row ``b`` maps logical
                cache blocks for active request ``b`` to physical block ids.
            seq_lens: [B], lengths already present before this invocation.
            position_ids: [Q], [1, Q], or [B, Q].  If omitted, absolute
                positions are derived from seq_lens.
            query_lens: [B], number of valid prefix tokens in each row of x.
                Defaults to Q for every active request.
            slot_mapping: Optional [B, Q] flattened physical cache slots.
                ``-1`` denotes an invalid/padded query.  If omitted, it is
                derived from block_tables and seq_lens.
        """

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [B, Q, d_model]")
        if self.training and torch.is_grad_enabled():
            raise RuntimeError(
                "paged attention mutates an inference-only K/V cache; "
                "call eval() before using it"
            )
        batch_size, query_length, _ = x.shape
        if batch_size <= 0 or query_length <= 0:
            raise ValueError("batch_size and query_length must be positive")
        if x.device != kv_cache.device or x.dtype != kv_cache.dtype:
            raise ValueError("x and the paged K/V cache must have the same device and dtype")
        if (kv_cache.n_kv_heads, kv_cache.head_dim) != (
            self.n_kv_heads,
            self.head_dim,
        ):
            raise ValueError("paged K/V cache shape does not match this attention layer")
        if seq_lens.shape != (batch_size,):
            raise ValueError("seq_lens must have shape [B]")
        if seq_lens.device != x.device:
            raise ValueError("seq_lens and x must be on the same device")
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("seq_lens must use torch.int32 or torch.int64")
        if (seq_lens < 0).any():
            raise ValueError("seq_lens cannot contain negative lengths")

        if query_lens is None:
            query_lens = torch.full(
                (batch_size,), query_length, device=x.device, dtype=torch.long
            )
        else:
            if query_lens.shape != (batch_size,):
                raise ValueError("query_lens must have shape [B]")
            if query_lens.device != x.device:
                raise ValueError("query_lens and x must be on the same device")
            if query_lens.dtype not in (torch.int32, torch.int64):
                raise TypeError("query_lens must use torch.int32 or torch.int64")
            if (query_lens < 0).any() or (query_lens > query_length).any():
                raise ValueError("query_lens values must be in [0, Q]")

        if position_ids is None:
            query_offsets = torch.arange(query_length, device=x.device, dtype=torch.long)
            position_ids = seq_lens.to(torch.long)[:, None] + query_offsets[None, :]
        elif position_ids.device != x.device:
            position_ids = position_ids.to(x.device)

        q, key, value = self._project_qkv(x, position_ids)

        if slot_mapping is None:
            slot_mapping = kv_cache.build_slot_mapping(
                block_tables=block_tables,
                start_positions=seq_lens,
                query_length=query_length,
                query_lens=query_lens,
            )
        else:
            if slot_mapping.shape != (batch_size, query_length):
                raise ValueError("slot_mapping must have shape [B, Q]")
            if slot_mapping.device != x.device:
                raise ValueError("slot_mapping and x must be on the same device")
            if slot_mapping.dtype not in (torch.int32, torch.int64):
                raise TypeError("slot_mapping must use torch.int32 or torch.int64")
            expected_valid = torch.arange(
                query_length, device=x.device
            )[None, :] < query_lens.to(torch.long)[:, None]
            if not (slot_mapping[expected_valid] >= 0).all() or not (
                slot_mapping[~expected_valid] == -1
            ).all():
                raise ValueError(
                    "slot_mapping must be non-negative for valid queries and -1 for padding"
                )
            expected_slots = kv_cache.build_slot_mapping(
                block_tables=block_tables,
                start_positions=seq_lens,
                query_length=query_length,
                query_lens=query_lens,
            )
            if not torch.equal(slot_mapping.to(torch.long), expected_slots):
                raise ValueError("slot_mapping is inconsistent with block_tables and seq_lens")

        kv_cache.write(key, value, slot_mapping)
        updated_seq_lens = seq_lens + query_lens.to(seq_lens.dtype)
        full_key, full_value, valid_keys = kv_cache.gather(block_tables, updated_seq_lens)
        output, valid_queries = self._compute_attention(
            q=q,
            key=full_key,
            value=full_value,
            valid_keys=valid_keys,
            past_seq_lens=seq_lens,
            query_lens=query_lens,
        )
        output = self.output_proj(output)
        # output_proj may have a bias, so clear padded query rows afterwards.
        output = output.masked_fill(~valid_queries[:, :, None], 0.0)
        return PagedAttentionOutput(output=output, seq_lens=updated_seq_lens)


class PagedMultiHeadAttention(BasePagedAttention):
    def __init__(self, d_model: int, config: AttentionConfig) -> None:
        if config.kind != "mha":
            raise ValueError(f"PagedMultiHeadAttention requires kind='mha', got {config.kind!r}")
        super().__init__(d_model, config, n_kv_heads=config.n_heads)


class PagedMultiQueryAttention(BasePagedAttention):
    def __init__(self, d_model: int, config: AttentionConfig) -> None:
        if config.kind != "mqa":
            raise ValueError(f"PagedMultiQueryAttention requires kind='mqa', got {config.kind!r}")
        super().__init__(d_model, config, n_kv_heads=1)


class PagedGroupQueryAttention(BasePagedAttention):
    def __init__(self, d_model: int, config: AttentionConfig) -> None:
        if config.kind != "gqa":
            raise ValueError(f"PagedGroupQueryAttention requires kind='gqa', got {config.kind!r}")
        if config.n_kv_heads is None:
            raise ValueError("PagedGroupQueryAttention requires n_kv_heads")
        super().__init__(d_model, config, n_kv_heads=config.n_kv_heads)


def build_paged_attention(
    d_model: int,
    config: AttentionConfig,
) -> PagedMultiHeadAttention | PagedMultiQueryAttention | PagedGroupQueryAttention:
    if config.kind == "mha":
        module = PagedMultiHeadAttention
    elif config.kind == "mqa":
        module = PagedMultiQueryAttention
    elif config.kind == "gqa":
        module = PagedGroupQueryAttention
    else:
        raise ValueError(f"paged attention does not support kind={config.kind!r}")
    return module(d_model, config)
