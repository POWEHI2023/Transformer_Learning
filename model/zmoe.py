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


@dataclass
class MoEConfig:
    expert_num: int
    top_k: int
    use_z_loss: bool = False # 限制 Router Logits 过快变大, 避免 softmax 接近 ont-host

    # 验证一下有效性, 避免 (expert_num=4, top_k=0) 之类的错误 config
    def __post_init__(self) -> None:
        if self.expert_num <= 0:
            raise ValueError("expert_num must be greater than zero")
        if not 1 <= self.top_k <= self.expert_num:
            raise ValueError("top_k must satisfy 1 <= top_k <= expert_num")

@dataclass
class RouterOutput:
    expert_indices: torch.Tensor    # [token_num, k_index]
    expert_weights: torch.Tensor     # [token_num, k_index]
    probabilities: torch.Tensor     # [token_num, expert_num]
    aux_loss: torch.Tensor
    z_loss: torch.Tensor | None
    stats: RouterStats

class TopKRouter(torch.nn.Module):
    def __init__(self, d_model: int, config: MoEConfig) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = config.top_k
        self.expert_num = config.expert_num
        self.use_z_loss = config.use_z_loss
        self.router = torch.nn.Linear(self.d_model, config.expert_num, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        '''input: hidden state for tokens, shape is [T, d_model]'''
        assert x.ndim == 2, "TopKRouter's input shape must be [T, d_model]"
        t, d_model = x.shape
        assert t > 0 and d_model > 0, \
            f"token_num and d_model must be greater than zero, but got (t: {t}, d_model: {d_model})"
        assert d_model == self.d_model, f"expected d_model={self.d_model}, got {d_model}"

        # [T, d_model] -> [T, expert_num]
        x = self.router(x).float()
        # [T, expert_num]
        probabilities = torch.softmax(x, dim=-1)
        # [T, expert_num] -> [T, K]
        # 避免 logits 偏差, 使用原来的 x 计算 top k
        topk_logits, expert_indices = x.topk(self.top_k, dim=-1)
        expert_weights = torch.softmax(topk_logits, dim=-1)

        # [T, expert_num] -> [expert_num]
        importance = probabilities.mean(0)
        # [E], [E] / T * K, T * K 表示总分配的 expert 数量
        # 计算每个专家分配的数量占总分配的多少
        tokens_per_expert = torch.bincount(expert_indices.reshape(-1), minlength=self.expert_num)
        load = tokens_per_expert.float() / (t * self.top_k)
        # L = E \sum_E importance(平均软概率) * load(实际硬路由比例)
        # 避免负载不平均的问题
        aux_loss = self.expert_num * torch.sum(importance * load)

        if self.use_z_loss:
            z_loss = torch.logsumexp(x, dim=-1).square().mean()
        else: z_loss = None

        return RouterOutput(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            probabilities=probabilities,
            aux_loss=aux_loss,
            z_loss=z_loss,
            stats=RouterStats(
                tokens_per_expert=tokens_per_expert.detach(),
                probability_per_expert=importance.detach(),
            ),
        )


class TopKSparseMoE(torch.nn.Module):
    pass
