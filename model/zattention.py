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