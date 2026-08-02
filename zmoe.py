# Prepare for MoE extension

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim or ((8 * self.d_model) // 3)

        self.gate_proj = torch.nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.a_up_proj = torch.nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.a_down_proj = torch.nn.Linear(self.hidden_dim, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, d_model] -> [B, L, hidden_dim]
        gate = F.silu(self.gate_proj(x))
        # [B, L, d_model] -> [B, L, hidden_dim]
        up = self.a_up_proj(x)
        # [B, L, hidden_dim] -> [B, L, d_model]
        return self.a_down_proj(gate * up)


@dataclass
class RouterStats:
    tokens_per_expert: torch.Tensor
    probability_per_expert: torch.Tensor


@dataclass
class FFNOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    router_stats: RouterStats | None


class DenseFFN(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
    # token_mask remained for MoE expansion

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> FFNOutput:
        x = self.dropout(self.swiglu(self.norm(x)))
        return FFNOutput(
            hidden_states=x,
            aux_loss=x.new_zeros((), dtype=torch.float32),
            router_stats=None,
        )
