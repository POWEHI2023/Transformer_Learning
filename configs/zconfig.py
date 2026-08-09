from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Any

import yaml

@dataclass(frozen=True)
class AttentionConfig:
    kind: Literal["mha", "mqa", "gqa", "mla"]
    backend: Literal["eager", "einsum", "sdpa", "flash"]
    n_heads: int
    n_kv_heads: int | None = None
    use_packed_segment: bool = False
    use_causal_mask: bool = True
    rope_base: float = 10_000.0

    def validate(self, d_model: int) -> None:
        if self.n_heads <= 0: raise ValueError("attention.n_heads must be positive")
        if d_model % self.n_heads != 0: raise ValueError("d_model must be divisible by n_heads")

        head_dim = d_model // self.n_heads
        if head_dim % 2 != 0: raise ValueError("head_dim must be even for RoPE")

        if self.kind == "mqa" and self.n_kv_heads not in (None, 1): raise ValueError("MQA requires n_kv_heads=1")

        if self.kind == "gqa":
            if self.n_kv_heads is None: raise ValueError("GQA requires n_kv_heads")
            if self.n_kv_heads <= 0: raise ValueError("attention.n_kv_heads must be positive")
            if self.n_heads % self.n_kv_heads != 0: raise ValueError("n_heads must be divisible by n_kv_heads")

@dataclass(frozen=True)
class MoEConfig:
    expert_num: int
    top_k: int
    use_z_loss: bool = False

    def validate(self) -> None:
        if self.expert_num <= 1: raise ValueError("expert_num must be greater than 1")
        if not 1 < self.top_k <= self.expert_num: raise ValueError("top_k must satisfy 1 < top_k <= expert_num")

@dataclass(frozen=True)
class GEMMConfig:
    mode: Literal["batched", "grouped"] = "batched"

@dataclass(frozen=True)
class FFNConfig:
    kind: Literal["dense", "moe"]
    backend: Literal["eager", "gemm"]
    hidden_dim: int
    moe: MoEConfig | None = None
    gemm: GEMMConfig | None = None

    def validate(self) -> None:
        if self.hidden_dim <= 0: raise ValueError("ffn.hidden_dim must be positive")

        # 如果 DenseFFN 那么就只能设置 eager, 其他配置都不重要
        if self.kind == "dense":
            if self.backend != "eager":
                raise ValueError("DenseFFN only supports eager backend")
            return

        # 如果使用 MoE, 那么就必须传入 MoEConfig
        if self.moe is None:
            raise ValueError("MoE FFN requires ffn.moe")
        self.moe.validate()

        # 如果 MoE 使用 GEMM 后端, 就需要 GEMMConfig
        if self.backend == "gemm":
            if self.gemm is None: raise ValueError("GEMM MoE requires ffn.gemm")

# 当前的 Config 中主要包含 AttentionConfig 和 FFNConfig,
# TODO 后续可能扩展加入 ParallelConfig,
# TODO 增加工厂创建 Attention 和 FFN, 实现工厂前需要统一所有 Attention 的输入和输出, FFN 已经完成统一

@dataclass(frozen=True)
class ParallelConfig:
    '''仅作为占位使用, 目前还没有 Parallel 的实现支持'''
    strategy: str = "single"
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    sequence_parallel: bool = False
    communication_backend: str = "nccl"

@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    d_model: int
    dropout: float
    # 表示让输入词嵌入和输出语言模型分类头共享同一份权重.
    # 输入和输出使用相同的 token 表示空间, 通常具有一定正则化效果，可能改善语言模型效果.
    # 两条路径产生的梯度会共同更新这份权重, 代价是输入表示和输出分类权重不能独立学习.
    tie_embedding: bool

    attention: AttentionConfig
    ffn: FFNConfig

    def validate(self) -> None:
        if self.n_layers <= 0: raise ValueError("model.n_layers must be positive")
        self.attention.validate(self.d_model)
        self.ffn.validate()

def load_model_config(config_path: str | Path) -> ModelConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(...)

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML config: {path}") from error

def model_config_from_dict(
    model_data: Mapping[str, Any],
    parallel_data: Mapping[str, Any] | None = None,
) -> ModelConfig:
    ...

def build_model_config(args: argparse.Namespace) -> ModelConfig:
    n_kv_heads = (
        None
        if args.attention_kind == "mha"
        else 1
        if args.attention_kind == "mqa"
        else args.n_kv_heads
    )
    attention_config = AttentionConfig(
        kind=args.attention_kind,
        backend=args.attention_backend,
        n_heads=args.heads,
        n_kv_heads=n_kv_heads,
        use_packed_segment=False,
        use_causal_mask=True,
        rope_base=10_000.0,
    )

    moe_config = (
        MoEConfig(
            expert_num=args.expert_num,
            top_k=args.top_k,
            use_z_loss=args.use_moe_z_loss,
        )
        if args.use_moe
        else None
    )
    ffn_config = FFNConfig(
        kind="moe" if args.use_moe else "dense",
        backend=args.moe_backend if args.use_moe else "eager",
        hidden_dim=args.hidden_dim,
        moe=moe_config,
        gemm=(
            GEMMConfig(mode=args.gemm_mode)
            if args.use_moe and args.moe_backend == "gemm"
            else None
        ),
    )
    model_config = ModelConfig(
        n_layers=args.layers,
        d_model=args.d_model,
        dropout=args.dropout,
        tie_embedding=args.tie_embedding,
        attention=attention_config,
        ffn=ffn_config,
    )
    model_config.validate()
    return model_config