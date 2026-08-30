from __future__ import annotations

import math
import torch
import torch.distributed as dist
from dataclasses import dataclass

from model.position.zposition import RoPE

@dataclass
class AttentionConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int | None = None
    use_packed_segment: bool = False
    use_causal_mask: bool = True
    partial_output: bool = False
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        for name, value in {"d_model": self.d_model, "n_heads": self.n_heads}.items():
            if value <= 0: raise ValueError(f"{name} must be a positive integer")
        if self.n_kv_heads is not None and self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be None or a positive integer")
        if (
            not isinstance(self.rope_base, (int, float))
            or isinstance(self.rope_base, bool)
            or not math.isfinite(self.rope_base)
            or self.rope_base <= 0
        ): raise ValueError("rope_base must be a finite positive number")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        n_kv_heads = (
            self.n_kv_heads
            if self.n_kv_heads is not None
            else self.n_heads
        )
        if self.n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not isinstance(self.partial_output, bool):
            raise TypeError("partial_output must be a bool")

@dataclass
class ParallelContext:
    tp_group: dist.ProcessGroup
    cp_group: dist.ProcessGroup
    replica_group: dist.ProcessGroup # 固定 TP, 包含 DP x CP

    tp_rank: int
    cp_rank: int
    dp_rank: int

@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    communication_backend: str = "nccl"

    _context: ParallelContext | None = None

    def __post_init__(self) -> None:
        parallel_sizes = {
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }
        for name, size in parallel_sizes.items():
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.sequence_parallel, bool):
            raise TypeError("sequence_parallel must be a bool")
        if (
            not isinstance(self.communication_backend, str)
            or not self.communication_backend.strip()
        ): raise ValueError("communication_backend must be a non-empty string")

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

@dataclass
class AttentionOutput:
    output: torch.Tensor
    cached_key_value: tuple[torch.Tensor, torch.Tensor] | None = None

class ParallelAttention(torch.nn.Module):
    def __init__(
        self,
        config: AttentionConfig,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.n_qo_heads = config.n_heads # 全局 query & output 的 Head 个数
        self.n_kv_heads = (
            config.n_kv_heads 
            if config.n_kv_heads is not None
            else config.n_heads 
        ) # 全局 key & value 的 Head 个数
        self.head_dim = self.d_model // self.n_qo_heads
        # group 个数为 n_kv_heads
        self.qo_heads_per_group = self.n_qo_heads // self.n_kv_heads # 全局 每个 Group 中的 query & output head 个数

        # ---------
        self.config = config
        self.parallel_config = (
            parallel_config
            if parallel_config is not None
            else ParallelConfig()
        )

        tp_size = self.parallel_config.tensor_parallel_size
        if self.n_qo_heads % tp_size != 0:
            raise ValueError("n_heads must be divisible by TP size")
        self.local_qo_heads = self.n_qo_heads // tp_size # 每个 TP 分片中的 query output head 个数

        if self.n_kv_heads == 1:
            self.local_kv_heads = self.n_kv_heads
            self._kv_tp_mode = "replicated"
        else:
            if self.n_kv_heads % tp_size != 0:
                raise ValueError("MHA/GQA require n_kv_heads to be divisible by TP size")
            self.local_kv_heads = self.n_kv_heads // tp_size
            self._kv_tp_mode = "shard"

        if self.local_qo_heads % self.local_kv_heads != 0:
            raise ValueError("local Q heads must be divisible by local KV heads")
        self.local_qo_heads_per_group = self.local_qo_heads // self.local_kv_heads
        if self._kv_tp_mode == "shard" and self.local_qo_heads_per_group != self.qo_heads_per_group:
            raise ValueError("TP sharding breaks GQA group boundaries")

        expected_world_size = self.parallel_config.world_size
        if expected_world_size > 1:
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized when parallel size is greater than 1")
            if dist.get_world_size() != expected_world_size:
                raise ValueError(
                    f"world_size={dist.get_world_size()} does not match "
                    f"DP×CP×TP×PP={expected_world_size}"
                )

        # ----------
        if self.n_kv_heads == self.n_qo_heads:
            self.attention_kind = "mha"
        elif self.n_kv_heads == 1:
            self.attention_kind = "mqa"
        else:
            self.attention_kind = "gqa"

        self._init_model()

    def _init_model(self) -> None:
        self.query_proj = torch.nn.Linear(self.d_model, self.local_qo_heads * self.head_dim)
        self.key_proj = torch.nn.Linear(self.d_model, self.local_kv_heads * self.head_dim)
        self.value_proj = torch.nn.Linear(self.d_model, self.local_kv_heads * self.head_dim)

        self.rope = RoPE(head_dim=self.head_dim, base=self.config.rope_base)

        self.output_proj = torch.nn.Linear(self.local_qo_heads * self.head_dim, self.d_model, bias=False)

    """
    需要三个支持反向传播的通信原语:
    copy_to_tp: 输入 x 在 TP rank 之间复制, 每个 rank 计算一部分 heads, 反向时需要聚合各 head 分片对 x 的梯度.
        forward: identity, backward: TP all_redule(sum)
    reduce_from_tp: 每个 TP rank 的 output_proj 只产生一部分结果, 需要把结果相加后才是完整输出.
        forward: TP all_reduce(sum), backward: identity
    _GatherFromCP.apply: forward: all_gather, backward: reduce_scatter
    """
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor, # 当前 x 中 sequence 所处的位置
        attention_mask: torch.Tensor | None = None,
        *,
        segment_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        B, S_local, d_model = x.shape
        if use_cache or past_key_value is not None:
            raise RuntimeError("Distributed KV Cache is not supported yet.")
        if self.config.use_packed_segment or segment_ids is not None:
            raise NotImplementedError("Do not support packed segment yet.")
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).to(x.device)
        else: position_ids = position_ids.to(x.device)
        if position_ids.shape not in {(1, S_local), (B, S_local)}:
            raise ValueError(
                "position_ids must have shape "
                f"[{S_local}], [1, {S_local}] or [{B}, {S_local}]"
            )
        attention_mask = (
            attention_mask.to(device=x.device, dtype=torch.bool)
            if attention_mask is not None
            else torch.ones(B, S_local, dtype=torch.bool, device=x.device) # 全部允许
        )
        if attention_mask.shape != (B, S_local):
            raise ValueError(f"attention shape must match ({B}, {S_local}), but got {attention_mask.shape}")

        """
        CP 开启时: S_local == S_global // cp
            x: [B, S_local, d_model]
            position:ids: [B, S_local]
            attention_mask: [B, S_local]
            segment_ids: [B, S_local]
        """
        if d_model != self.d_model:
            raise ValueError("Input model dimension does not match Attention Layer model dimension.")

        parallel_context = self.parallel_config._context

        if self.parallel_config.tensor_parallel_size > 1:
            if parallel_context is None:
                raise RuntimeError("ParallelContext is required when TP is enabled")
            x_for_tp = _CopyToTP.apply(x, parallel_context.tp_group)
        else:
            x_for_tp = x

        q = self.query_proj(x_for_tp)
        k = self.key_proj(x_for_tp)
        v = self.value_proj(x_for_tp)

        q = q.view(B, S_local, self.local_qo_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S_local, self.local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_local, self.local_kv_heads, self.head_dim).transpose(1, 2)

        # 每个 CP rank 使用自己的全局位置, RoPE 不需要修改
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        if self.parallel_config.context_parallel_size > 1:
            if parallel_context is None:
                raise RuntimeError("ParallelContext is required when CP is enabled")

            # 先直接拷贝所有 KV 到 Local 后续再考虑 Ring Attention 实现
            k_global = _GatherFromCP.apply(k, 2, parallel_context.cp_group)
            v_global = _GatherFromCP.apply(v, 2, parallel_context.cp_group)

            key_position_ids = gather_metadata_from_cp(
                position_ids, dim=1,
                group=parallel_context.cp_group,
            )
            key_attention_mask = gather_metadata_from_cp(
                attention_mask, dim=1,
                group=parallel_context.cp_group,
            )
            # key_segment_ids = gather_metadata_from_cp(segment_ids)
        else:
            # CP == 1
            k_global, v_global = k, v
            key_position_ids, key_attention_mask = position_ids, attention_mask
            
        # x: [B, Hq_local, S_local, D]
        # k: [B, Hq_local, S_global, D]
        # v: [B, Hq_local, S_global, D]
        repeat_factor = self.local_qo_heads // self.local_kv_heads
        k_for_q = k_global.repeat_interleave(repeat_factor, dim=1)
        v_for_q = v_global.repeat_interleave(repeat_factor, dim=1)

        # [B, Hq_local, S_local, S_global]
        score = torch.matmul(
            q.float(),
            k_for_q.float().transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        # CP 情况下必须使用全局 causal mask
        causal_blocked = (
            key_position_ids[:, None, None, :] # [B, S_global]
            > position_ids[:, None, :, None] # [B, S_local]
        ) # [B, 1, S_local, S_global]

        padding_blocked = ~key_attention_mask[:, None, None, :].bool()
        if self.config.use_packed_segment and segment_ids is not None:
            segment_blocked = (
                segment_ids[:, None, :, None]
                != key_segment_ids[:, None, None, :]
            ) # 使用 packed segment 场景下使用的 mask, 避免不同 segment 之间相互干扰
        else: segment_blocked = None

        blocked_mask = None
        if self.config.use_causal_mask:
            blocked_mask = causal_blocked
        blocked_mask = (
            padding_blocked
            if blocked_mask is None
            else blocked_mask | padding_blocked
        )
        if segment_blocked is not None:
            blocked_mask = blocked_mask | segment_blocked

        score = score.masked_fill(blocked_mask, float("-inf"))

        probs = torch.softmax(score, dim=-1, dtype=torch.float32)
        probs = probs.to(v_for_q.dtype)

        output = probs @ v_for_q # [B, Hq_local, S_local, D]

        # 合并本地 heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, S_local, self.local_qo_heads * self.head_dim)
        partial_output = self.output_proj(output)

        """
        串行 Attention 之后 FFN 计算时必须执行一次通信聚合结果,
        可以通过 SP(复用 TP 的分片配置) 在执行 Reduce Scatter, 将 Sequence 分片 [B, S_cp/TP, d_model],
        -> Residual 和 Norm 期间 activation 保持 sequence-sharded; 降低显存; 更容易重叠通信与计算
        """
        if self.parallel_config.tensor_parallel_size > 1 and not self.config.partial_output:
            if parallel_context is None:
                raise RuntimeError("ParallelContext is required when TP is enabled")
            # 这个输出中 CP 继续分片, 只聚合 TP attention 的输出
            # 放在 output_proj 之后进行 reduce sum, 如果在 proj 之前就需要 gather
            output = _ReduceFromTP.apply(partial_output, parallel_context.tp_group)
        else:
            # TP > 1 时保留各 TP rank 的 Partial(SUM) 输出，由上层组合后统一聚合
            output = partial_output
        
        return AttentionOutput(
            output=output,
            cached_key_value=None, # 目前还不支持 Decode
        )

"""
forward: identity, 假设 TP rank 拥有相同的输入
backward: all_reduce(sum) 所有分片之间的梯度
"""
class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return x
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_input = grad_output.clone()
        dist.all_reduce(
            grad_input,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        # group 不需要 grade
        return grad_input, None

"""
forward: 收集 CP 中所有 Sequence 的 KV
backward: CP reduce_scatter(sum), 用到了其他分片的 KV, 必须把不同 CP rank 对同一个 KV chunk 的梯度求和
"""
class _GatherFromCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
        group_size = dist.get_world_size(group)
        if dim < 0:
            dim += x.ndim
        if not 0 <= dim < x.ndim:
            raise ValueError(
                f"Invalid gather dimension {dim} for tensor with "
                f"{x.ndim} dimensions"
            )

        x_contiguous = x.contiguous()
        gathered = [
            torch.empty_like(x_contiguous)
            for _ in range(group_size)
        ]
        dist.all_gather(
            gathered,
            x_contiguous,
            group=group,
        )

        ctx.group = group
        ctx.group_size = group_size
        ctx.dim = dim
        ctx.local_dim_size = x.shape[dim]

        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        expected_size = (ctx.local_dim_size * ctx.group_size)
        if grad_output.shape[ctx.dim] != expected_size:
            raise RuntimeError(
                "Global gradient dimension does not match "
                "the CP gather layout"
            )
        grad_chunks = [
            chunk.contiguous()
            for chunk in torch.split(
                grad_output,
                ctx.local_dim_size,
                dim=ctx.dim,
            )
        ]

        if len(grad_chunks) != ctx.group_size:
            raise RuntimeError(
                f"Expected {ctx.group_size} gradient chunks, "
                f"but got {len(grad_chunks)}"
            )
        grad_input = torch.empty_like(grad_chunks[0])
        dist.reduce_scatter(
            grad_input,
            grad_chunks,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )

        return grad_input, None, None

"""
forward: 收集 CP 之间的 position_ids, attention_mask, segment_ids
backward: 不参与
"""
@torch.no_grad()
def gather_metadata_from_cp(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    if dim < 0:
        dim += x.ndim
    if not 0 <= dim < x.ndim:
        raise ValueError(
            f"Invalid gather dimension {dim} "
            f"for tensor with {x.ndim} dimensions"
        )

    group_size = dist.get_world_size(group)
    if group_size == 1:
        return x

    x_contiguous = x.contiguous()
    gathered = [
        torch.empty_like(x_contiguous)
        for _ in range(group_size)
    ]
    dist.all_gather(
        gathered,
        x_contiguous,
        group=group,
    )

    # 在 Sequence 维度上聚合
    return torch.cat(gathered, dim=dim)

"""
forward: TP all_reduce(sum)
backward: identity
"""
class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        output = x.clone()
        dist.all_reduce(
            output,
            op=dist.ReduceOp.SUM,
            group=group,
        )
        return output
    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None]:
        return grad_output, None
