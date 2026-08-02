import torch
import math
from typing import Any
import torch.nn.functional as F
from dataclasses import dataclass
from zmoe import RouterStats, FFNOutput, DenseFFN

class RoPE(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        # inv_freq_i = 1 / base ^ { 2i / head_dim }, [head_dim/2]
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, head_dim]/ [L]
        batch_size, _, sequence_length, _ = x.shape

        if position_ids.ndim == 1:
            # [L]
            assert position_ids.shape == (sequence_length,), "1D position_ids must have shape [L]"
            # [1, L]
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.ndim == 2:
            assert position_ids.shape in (
                (1, sequence_length),
                (batch_size, sequence_length),
            ), "2D position_ids must have shape [1, L] or [B, L]"
        else:
            raise ValueError("position_ids must have shape [L] or [B, L]")
        # [B, L]
        position_ids = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)

        # position_ids:     [B or 1, L] -> [B or 1, L, 1]
        # inv_freq:         [head_dim/2] -> [1, 1, head_dim/2]
        # broadcast angles: [B or 1, L, head_dim/2]
        angles = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        # [B or 1, 1, L, head_dim/2]
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # [B, H, L, head_dim/2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # [B, H, L, head_dim/2, 2]
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        # [B, H, L, head_dim]
        return rotated.flatten(-2)

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
    ) -> torch.Tensor:
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

        return self.output_proj(o)
    

@dataclass
class TransformerBlockOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    router_stats: RouterStats | None

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int = 512,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.attn_norm = torch.nn.LayerNorm(d_model)
        self.attn = Attention(n_heads=n_heads, d_model=d_model, use_causal_mask=use_causal_mask)    
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.ffn = DenseFFN(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> TransformerBlockOutput:
        # in: [B, L, d_model], out: [B, L, d_model]
        y = self.attn_norm(x)
        y = self.attn(y, position_ids, attention_mask)
        y = self.attn_dropout(y)
        x = x + y

        ffn_o: FFNOutput = self.ffn(x, token_mask=attention_mask)
        return TransformerBlockOutput(
            hidden_states=x + ffn_o.hidden_states,
            aux_loss=ffn_o.aux_loss,
            router_stats=ffn_o.router_stats,
        )

@dataclass
class TransformerOutput:
    logits: torch.Tensor
    router_aux_loss: torch.Tensor
    router_stats: list[RouterStats]

class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_heads: int = 2,
        d_model: int = 512,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        tie_embedding: bool = True,
        pad_token_id: int | None = None,
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
            ) for _ in range(n_layers)
        ])

        self.final_norm = torch.nn.LayerNorm(d_model)
        # [B, L, d_model] -> [B, L, vocab_size]
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        # share weight
        if tie_embedding and self.pad_token_id is None:
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
        router_stats: list[RouterStats] = []

        # [B, L, d_model]
        for layer in self.layers:
            # Outer position ids
            block_o: TransformerBlockOutput = layer(x, position_ids, attention_mask)
            x = block_o.hidden_states
            if block_o.router_stats is not None:
                aux_losses.append(block_o.aux_loss)
                router_stats.append(block_o.router_stats)

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