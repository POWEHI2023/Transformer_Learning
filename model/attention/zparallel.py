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

@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    communication_backend: str = "nccl"

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

class Attention(torch.nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        if use_cache or past_key_value is not None:
            raise RuntimeError("Distributed KV Cache is not supported yet.")
        # TODO 继续实现...
        raise NotImplementedError()