from __future__ import annotations

import torch

from configs.zconfig import ModelConfig


def log_configs(
    *,
    model_config: ModelConfig,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    context_length: int,
    learning_rate: float,
    trainable_parameters: int,
    moe_aux_loss_coeff: float,
    moe_z_loss_coeff: float,
) -> None:
    print(
        f"training_config: device={device}, epochs={epochs}, "
        f"steps_per_epoch={steps_per_epoch}, batch_size={batch_size}, "
        f"context_length={context_length}, lr={learning_rate}"
    )
    print(
        f"model_config: layers={model_config.n_layers}, d_model={model_config.d_model}, "
        f"dropout={model_config.dropout}, tie_embedding={model_config.tie_embedding}, "
        f"trainable_parameters={trainable_parameters:,}"
    )

    attention_config = model_config.attention
    print(
        f"attention_config: kind={attention_config.kind}, backend={attention_config.backend}, "
        f"heads={attention_config.n_heads}, kv_heads={attention_config.n_kv_heads}, "
        f"causal={attention_config.use_causal_mask}, "
        f"packed_segment={attention_config.use_packed_segment}, rope_base={attention_config.rope_base}"
    )

    ffn_config = model_config.ffn
    print(
        f"ffn_config: kind={ffn_config.kind}, backend={ffn_config.backend}, "
        f"hidden_dim={ffn_config.hidden_dim}"
    )
    if ffn_config.moe is None:
        print("moe_config: disabled (using DenseFFN)")
        return

    moe_config = ffn_config.moe
    print(
        f"moe_config: experts={moe_config.expert_num}, top_k={moe_config.top_k}, "
        f"use_z_loss={moe_config.use_z_loss}, "
        f"aux_coeff={moe_aux_loss_coeff}, z_coeff={moe_z_loss_coeff}"
    )
    if ffn_config.gemm is not None:
        print(f"gemm_config: mode={ffn_config.gemm.mode}")
