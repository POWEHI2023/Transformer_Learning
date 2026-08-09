from __future__ import annotations

import torch
from dataclasses import dataclass
from model.zmoe import RouterStats, FFNOutput, DenseFFN, TopKSparseMoE, MoEConfig, GEMM_TopKSparseMoE
from model.zattention import MultiHeadAttention, AttentionOutput
from configs.zconfig import AttentionConfig


@dataclass
class TransformerBlockOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    z_loss: torch.Tensor | None
    router_stats: RouterStats | None

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_dim: int,
        d_model: int = 512,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        moe_config: MoEConfig | None = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.attn_norm = torch.nn.LayerNorm(d_model)
        _attn_config = AttentionConfig(
            kind="mha",
            backend="eager",
            n_heads=n_heads,
            n_kv_heads=None,
            use_packed_segment=False,
            use_causal_mask=use_causal_mask,
            rope_base=10_000.0,
        )
        self.attn = MultiHeadAttention(d_model=d_model, config=_attn_config)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.ffn = (
                GEMM_TopKSparseMoE(
                    d_model,
                    config=moe_config,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_grouped_gemm=False,
                )
                if moe_config is not None
                else DenseFFN(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> TransformerBlockOutput:
        # in: [B, L, d_model], out: [B, L, d_model]
        y = self.attn_norm(x)
        y: AttentionOutput = self.attn(y, position_ids, attention_mask)
        y = self.attn_dropout(y.output)
        x = x + y

        ffn_o: FFNOutput = self.ffn(x, token_mask=attention_mask)
        return TransformerBlockOutput(
            hidden_states=x + ffn_o.hidden_states,
            aux_loss=ffn_o.aux_loss,
            z_loss=ffn_o.z_loss,
            router_stats=ffn_o.router_stats,
        )

@dataclass
class TransformerOutput:
    logits: torch.Tensor
    router_aux_loss: torch.Tensor
    router_z_loss: torch.Tensor
    router_stats: list[RouterStats]

class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        hidden_dim: int,
        n_heads: int = 4,
        d_model: int = 512,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        tie_embedding: bool = True,
        pad_token_id: int | None = None,
        moe_config: MoEConfig | None = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0

        self.pad_token_id = pad_token_id
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            padding_idx=pad_token_id,
        )
        self.embedding_dropout = torch.nn.Dropout(dropout)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                n_heads=n_heads,
                d_model=d_model,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_causal_mask=use_causal_mask,
                moe_config=moe_config,
            ) for _ in range(n_layers)
        ])

        self.final_norm = torch.nn.LayerNorm(d_model)
        # [B, L, d_model] -> [B, L, vocab_size]
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        # share weight
        self.tie_embedding = tie_embedding
        # if tie_embedding and self.pad_token_id is None:
        if self.tie_embedding:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, 
        token_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> TransformerOutput:
        # [B, L]
        assert token_ids.dim() == 2 and token_ids.dtype == torch.long
        batch_size, sequence_length = token_ids.shape

        x = self.token_embedding(token_ids)
        x = self.embedding_dropout(x)

        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = token_ids != self.pad_token_id

        if attention_mask is not None:
            assert token_ids.shape == attention_mask.shape
            attention_mask = attention_mask.to(device=token_ids.device, dtype=torch.bool)
            assert attention_mask.any(dim=-1).all(), "Every sample must contain at least one valid token"

            padding_started = (~attention_mask).cummax(dim=-1).values
            has_valid_after_padding = (padding_started & attention_mask).any(dim=-1)
            assert not has_valid_after_padding.any().item(), \
                "Only right padding is supported. Expected masks like [1, 1, 1, 0, 0]."

        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=token_ids.device, dtype=torch.long)


        aux_losses: list[torch.Tensor] = []
        z_losses: list[torch.Tensor] = []
        router_stats: list[RouterStats] = []

        # [B, L, d_model]
        for layer in self.layers:
            # Outer position ids
            block_o: TransformerBlockOutput = layer(x, position_ids, attention_mask)
            x = block_o.hidden_states
            if block_o.router_stats is not None:
                aux_losses.append(block_o.aux_loss)
                router_stats.append(block_o.router_stats)
            if block_o.z_loss is not None:
                z_losses.append(block_o.z_loss)

        # [B, L, d_model]
        x = self.final_norm(x)
        # [B, L, d_model] -> [B, L, vocab_size]
        logits = self.lm_head(x)

        return TransformerOutput(
            logits=logits,
            router_aux_loss=(
                torch.stack(aux_losses).mean()
                if len(aux_losses) != 0
                else logits.new_zeros((), dtype=torch.float32)
            ),
            router_z_loss=(
                torch.stack(z_losses).mean()
                if len(z_losses) != 0
                else logits.new_zeros((), dtype=torch.float32)
            ),
            router_stats=router_stats,
        )


# Padding Mask: 屏蔽为了对齐长度而补充的 PAD token。
# Packed Sequence Mask: 防止拼接在同一序列中的不同样本互相关注, 为了避免浪费训练资源, 将多个短序列合并成一个长序列, (未实现)

# 左侧 Padding: PAD token 放在有效 token 的左侧, 有时用于批量自回归生成, 所有样本最后一个位置都是最新的 token。
# 右侧 Padding: PAD token 放在有效 token 的右侧, 右侧 Padding 是训练 Decoder-Only Transformer 时常见的方式, 有效 token 都位于序列开头, 不会出现有效 token 前面全是被屏蔽位置的情况。

# RoPE: 旋转位置编码, 

# TODO Packed segment attention mask; Packed segment boundary loss mask
# Packed sequence: 把多个独立样本拼进同一个长序列, 减少 padding 浪费.
# 1. Attention 层禁止 A, B 互相读取 - Packed segment attention mask
#       前样本的 token 序列虽然排在后样本的前面, 但不应该参与后样本的上下文, 需要构造 segment mask
# 2. Loss 层禁止 前序列被训练成预测后序列(不同样本) - Packed segment boundary loss mask
#       Attention Mask 只是限制了读取哪些 token 不会改变训练目标, 需要无效化 segment 边界对应的 label