# Prepare for MoE extension

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from abc import ABC, abstractmethod
from configs.zconfig import FFNConfig, MoEConfig


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

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
    z_loss: torch.Tensor | None
    router_stats: RouterStats | None


class DenseFFN(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
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
            z_loss=None,
            router_stats=None,
        )


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

@dataclass
class RoutingPlan:
    sorted_token_indices: torch.Tensor  # token, [R]
    sorted_expert_indices: torch.Tensor # expert, [R]
    sorted_weights: torch.Tensor        # weight, [R]
    tokens_per_expert: torch.Tensor     # length, [E]
    expert_offsets: torch.Tensor        # offsets, [E + 1]

    token_num: int

class BaseTopKSparseMoE(torch.nn.Module, ABC):
    def __init__(
        self, 
        d_model: int,
        config: MoEConfig,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = hidden_dim
        self.config = config

        # shared pre-norm
        self.norm = torch.nn.LayerNorm(d_model)
        # Top-K router
        self.router = TopKRouter(d_model=self.d_model, config=self.config)
        # shared dropout
        self.dropout = torch.nn.Dropout(dropout)

    @abstractmethod
    def forward(self,
        x: torch.Tensor, # [B, L, D]
        token_mask: torch.Tensor | None = None, # [B, L]
    ) -> FFNOutput:
        raise NotImplementedError

    def _combine_sorted_output(
        self,
        sorted_output: torch.Tensor,
        routing_plan: RoutingPlan,
    ) -> torch.Tensor:
        sorted_weights = routing_plan.sorted_weights.to(dtype=sorted_output.dtype)
        weighted_output = sorted_output * sorted_weights.unsqueeze(-1)

        valid_output = weighted_output.new_zeros(routing_plan.token_num, self.d_model)
        valid_output.index_add_(dim=0, index=routing_plan.sorted_token_indices, source=weighted_output)
        return valid_output

    def _build_routing_plan(
        self,
        valid_x: torch.Tensor,
        router_output: RouterOutput
    ) -> tuple[torch.Tensor, RoutingPlan]:
        '''拆分出来根据 valid_x 构造 RoutingPlan 的过程, 涉及到排序和选择性的 Padding Contiguous-Offset
        RoutingPlan 表示 sorted Expert-Major 布局逻辑'''
        token_num, top_k = router_output.expert_indices.shape
        route_token_indices = (
            torch.arange(token_num, device=valid_x.device)
            .unsqueeze(1)
            .expand(token_num, top_k)
            .reshape(-1)
        )
        route_expert_indices = router_output.expert_indices.reshape(-1)
        route_weights = router_output.expert_weights.reshape(-1)

        # 这里先把输入的 token 根据 expert id 排序,
        # 后续先还原 token major, 然后根据 valid_indices 还原 flat_output
        order = torch.argsort(route_expert_indices)
        sorted_expert_indices = route_expert_indices[order]
        sorted_token_indices = route_token_indices[order]
        sorted_weight = route_weights[order]

        # 由于[T, D] 根据 [T, K] 为 K 个 Expert Flatten 之后, 有[T * K] 个 token,
        # 其中存在重复的 token, sorted_x 的形状是 [T * K, D].
        sorted_x = valid_x.index_select(dim=0, index=sorted_token_indices)
        # 构造 Expert Offsets
        tokens_per_expert = \
            torch.bincount(sorted_expert_indices, minlength=self.config.expert_num)
        expert_offsets = torch.cat([
            tokens_per_expert.new_zeros(1),
            tokens_per_expert.cumsum(dim=0),
        ])

        return sorted_x, RoutingPlan(
            sorted_token_indices=sorted_token_indices,
            sorted_expert_indices=sorted_expert_indices,
            sorted_weights=sorted_weight,
            tokens_per_expert=tokens_per_expert,
            expert_offsets=expert_offsets,
            token_num=valid_x.shape[0],
        )


# 介绍一下 GEMM, 通用矩阵乘法, 如果在 for 循环中计算很多个 Expert 会导致启动很多个计算 Kernel
# 效率很差, 因此 Group GEMM 可以在 GPU Kernel 内部统一调度这些矩阵乘法提高效率,
# Batched GEMM 通常要求每组矩阵乘法形状相同， 所有的 Expert 必须有相同的 Token 数量,
# 但是 MoE 中每个 Expert 收到的 Token 数量通常不同, Grouped GEMM 可以处理不同的矩阵尺寸, 更适合 MoE.
# ---
# GEMM 是一次矩阵乘法；Grouped GEMM 是将多个 Expert 的、token 数可能不同的矩阵乘法统一提交和调度，从而减少小 kernel 启动开销并提高 GPU 利用率。
class TopKSparseMoE(BaseTopKSparseMoE):
    def __init__(
        self, 
        d_model: int,
        config: MoEConfig,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, config, hidden_dim, dropout)
        # experts
        self.experts = torch.nn.ModuleList([
            SwiGLU(d_model=self.d_model, hidden_dim=hidden_dim)
            for _ in range(config.expert_num)
        ])

    def forward(
        self,
        x: torch.Tensor, # [B, L, D]
        token_mask: torch.Tensor | None = None, # [B, L]
    ) -> FFNOutput:
        assert x.ndim == 3
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        # Token Mask
        token_mask = token_mask if token_mask is not None else torch.ones(batch_size, sequence_length)
        assert token_mask.shape == (batch_size, sequence_length)
        token_mask = token_mask.to(dtype=torch.bool, device=x.device)

        # Normalize and Flatten
        normalized_x = self.norm(x)
        # [B, L, D] -> [B * L, D]
        flat_x: torch.Tensor = normalized_x.reshape(batch_size * sequence_length, self.d_model)
        flat_mask: torch.Tensor = token_mask.reshape(-1)

        # 避免把 PAD token 送入 Router,
        # 影响 tokens_per_expert/probability_per_expert/auxiliary loss/z-loss
        valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            raise ValueError("TopKSparseMoE requires at least one valid token")
        valid_x = flat_x.index_select(dim=0, index=valid_indices) # [T, D]

        router_output: RouterOutput = self.router(valid_x)
        sorted_x, routing_plan = self._build_routing_plan(valid_x, router_output)

        # valid_output = self._dispatch_v1(valid_x=valid_x, router_output=router_output)
        valid_output = self._dispatch_v2(sorted_x=sorted_x, routing_plan=routing_plan)

        # valid_output 中只记录了 mask 之后不包含 PAD 的有效 token,
        # 需要把有效 token 映射回原本的总 token 空间中.
        flat_output = torch.zeros_like(flat_x)
        flat_output = flat_output.index_copy(dim=0, index=valid_indices, source=valid_output)

        output = flat_output.reshape(batch_size, sequence_length, self.d_model)
        output = self.dropout(output)

        # Token FFN 结果和 Router 的状态和损失
        return FFNOutput(
            hidden_states=output,
            aux_loss=router_output.aux_loss,
            z_loss=router_output.z_loss,
            router_stats=router_output.stats,
        )

    def _dispatch_v1(
        self,
        valid_x: torch.Tensor,
        router_output: RouterOutput,
    ) -> torch.Tensor:
        # build buffer for experts' output, shape is [T, D]
        # 累加所有 Expert 的加权结果
        valid_output = torch.zeros_like(valid_x)
        for expert_index, expert in enumerate(self.experts):
            # 每个 Token 的 K 个 Expert ID 中是否存在等于 expert_index 的 ID
            selected = (router_output.expert_indices == expert_index)
            # 把 Expert ID 中存在 expert_index 的 token id 返回,
            # route_slots 表示 Top K 中的第几个槽位选择当前 Expert.
            token_indices, route_slots = selected.nonzero(as_tuple=True)

            if token_indices.numel() == 0:
                # 没有 token 选择当前 expert
                continue

            # T 维根据选择的 token index 取出 token, 形状为 [T_e, D]
            expert_input = valid_x.index_select(dim=0, index=token_indices)
            # SwiGLU: input[T, D] -> [T, D]
            expert_output: torch.Tensor = expert(expert_input)

            # 加权记录结果, 每个 index 处的 slot 记录了 token 选择 Top-K 中当前 expert 占有的权重
            # shape is: [T_e]
            weights = router_output.expert_weights[token_indices, route_slots]
            weights = weights.to(dtype=expert_output.dtype)
            # [T_e, D] * [T_e, 1] -> 广播计算 D 维度乘同一个 weight [T_e, D]
            weighted_output = expert_output * weights.unsqueeze(-1)

            # 缓存加权结果: valid_output[token_indices[i]] += weighted_output[i]
            valid_output.index_add_(dim=0, index=token_indices, source=weighted_output)
        return valid_output

    def _dispatch_v2(
        self,
        sorted_x: torch.Tensor,
        routing_plan: RoutingPlan,
    ) -> torch.Tensor:
        sorted_output = torch.zeros_like(sorted_x)
        for expert_index, expert in enumerate(self.experts):
            start = routing_plan.expert_offsets[expert_index]
            end = routing_plan.expert_offsets[expert_index + 1]

            expert_input = sorted_x[start : end]
            if expert_input.shape[0] == 0:
                continue
            sorted_output[start : end] = expert(expert_input)

        sorted_weights = routing_plan.sorted_weights.to(dtype=sorted_output.dtype)
        # [T * K, D], 根据 Expert 排列的
        weighted_output = sorted_output * sorted_weights.unsqueeze(-1)
        _tk, _d = weighted_output.shape
        assert _tk // self.config.top_k == routing_plan.token_num
        # 从 routing plan 还原 order
        valid_output = torch.zeros(
            (routing_plan.token_num, _d),
            dtype=weighted_output.dtype,
            device=weighted_output.device,
        )
        # [T * K, D] -> [T, D]
        valid_output.index_add_(dim=0, index=routing_plan.sorted_token_indices, source=weighted_output)
        return valid_output
        
class GEMM_TopKSparseMoE(BaseTopKSparseMoE):
    def __init__(
        self, 
        d_model: int,
        config: MoEConfig,
        hidden_dim: int,
        dropout: float = 0.0,
        use_grouped_gemm: bool = True,
    ) -> None:
        super().__init__(d_model, config, hidden_dim, dropout)
        self.use_grouped_gemm = use_grouped_gemm
        # Packed Expert Weights
        self.gate_up_weight = torch.nn.Parameter(
            torch.empty(config.expert_num, d_model, 2 * hidden_dim)
        )
        self.down_weight = torch.nn.Parameter(
            torch.empty(config.expert_num, hidden_dim, d_model)
        )
        self._reset_parameters() # 初始化权重参数

    def _reset_parameters(self) -> None:
        gate_up_bound = 1.0 / math.sqrt(self.d_model)
        down_bound = 1.0 / math.sqrt(self.d_hidden)
        torch.nn.init.uniform_(self.gate_up_weight, -gate_up_bound, gate_up_bound)
        torch.nn.init.uniform_(self.down_weight, -down_bound, down_bound)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> FFNOutput:
        assert x.ndim == 3
        batch_size, sequence_length, d_model = x.shape
        assert d_model == self.d_model

        # Token Mask
        token_mask = token_mask if token_mask is not None else torch.ones(batch_size, sequence_length)
        assert token_mask.shape == (batch_size, sequence_length)
        token_mask = token_mask.to(dtype=torch.bool, device=x.device)

        # Normalize and Flatten
        normalized_x = self.norm(x)
        flat_x: torch.Tensor = normalized_x.reshape(batch_size * sequence_length, self.d_model)
        flat_mask: torch.Tensor = token_mask.reshape(-1)

        valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            raise ValueError("TopKSparseMoE requires at least one valid token")
        valid_x = flat_x.index_select(dim=0, index=valid_indices)

        router_output: RouterOutput = self.router(valid_x)
        # 当前每个 Expert 中包含的 token 数量不同, token_per_expert 不同
        sorted_x, routing_plan = self._build_routing_plan(valid_x, router_output)

        sorted_output = (
            self._experts_batched(sorted_x, routing_plan) # [T * K, D]
            if not self.use_grouped_gemm
            else self._experts_grouped(sorted_x, routing_plan)
        )

        valid_output = self._combine_sorted_output(sorted_output, routing_plan) # [T, D]

        flat_output = torch.zeros_like(flat_x)
        flat_output = flat_output.index_copy(dim=0, index=valid_indices, source=valid_output)
        output = flat_output.reshape(batch_size, sequence_length, self.d_model)
        output = self.dropout(output)

        return FFNOutput(
            hidden_states=output,
            aux_loss=router_output.aux_loss,
            z_loss=router_output.z_loss,
            router_stats=router_output.stats,
        )

    def _experts_batched(self, sorted_x: torch.Tensor, routing_plan: RoutingPlan) -> torch.Tensor:
        # 取出单个 expert 消费最多的 token 数量
        capacity = routing_plan.tokens_per_expert.max().item()

        # 每个 Token 有 Top-K 条路由, 总的路由数量为 T * K
        route_num = sorted_x.shape[0]
        expert_ids = routing_plan.sorted_expert_indices
        # 先获得每个 Expert 排序之后第一次出现的位置
        expert_start_per_route = (
            routing_plan.expert_offsets[:-1]
            .index_select(dim=0, index=expert_ids)
        )
        # 然后为发送给 Expert 的 Token 设置一个位置, 先是在 Expert 内部的位置
        local_positions = ( # [T * K]
            torch.arange(route_num, device=sorted_x.device)
            - expert_start_per_route
        )
        # 然后根据 Expert 第一次出现的位置和 Token 在 Expert 内的位置, 把位置广播到全局
        # 对应 [E, C, D] -> 第几个 Expert, 每个 Expert 有 Capacity 个 Token, 选择第几个 Token
        flat_positions = (expert_ids * capacity) + local_positions

        # 构造 Padded X, [E * C, D]
        padded_flat: torch.Tensor = sorted_x.new_zeros((self.config.expert_num * capacity, self.d_model))
        # 根据 flat_positions 把 sorted_x 中的 token 放到 padded_flat 中
        padded_flat = padded_flat.index_copy_(dim=0, index=flat_positions, source=sorted_x)
        # [E * C, D] -> [E, C, D]
        padded_x = padded_flat.reshape(self.config.expert_num, capacity, self.d_model)
        # 按照 SwiGLU 的实现, 计算 gate 和 up, 然后 down(silu(gate) * up)
        gate_up = torch.bmm(padded_x, self.gate_up_weight)  # [E, C, 2H], gate & up
        gate, up = gate_up.split([self.d_hidden, self.d_hidden], dim=-1)
        hidden = F.silu(gate) * up # [E, C, H]
        padded_output = torch.bmm(hidden, self.down_weight) # [E, C, D]

        # 删去 Padded Token, 取出有效 Token
        sorted_output = (
            padded_output
            .reshape(self.config.expert_num * capacity, self.d_model)
            .index_select(dim=0, index=flat_positions)
        )
        return sorted_output

    def _experts_grouped(self, sorted_x: torch.Tensor, routing_plan: RoutingPlan) -> torch.Tensor:
        # 使用结束位置, [E]
        offs = routing_plan.expert_offsets[1:].to(dtype=torch.int32, device=sorted_x.device)

        # [T * K, D] [E, D, 2H] -> [T * K, 2H]
        gate_up = F.grouped_mm(sorted_x, self.gate_up_weight, offs=offs)
        gate, up = gate_up.split(self.d_hidden, dim=-1)
        hidden = F.silu(gate) * up

        # [T * K, H] [E, H, D] -> [T * K, D]
        sorted_output = F.grouped_mm(hidden, self.down_weight, offs=offs)
        return sorted_output

@torch.no_grad()
def load_gemm_from_eager(gemm_moe: GEMM_TopKSparseMoE, eager_moe: TopKSparseMoE) -> None:
    assert eager_moe.config.expert_num == gemm_moe.config.expert_num
    assert eager_moe.d_model == gemm_moe.d_model
    assert eager_moe.d_hidden == gemm_moe.d_hidden

    gemm_moe.norm.load_state_dict(eager_moe.norm.state_dict())
    gemm_moe.router.load_state_dict(eager_moe.router.state_dict())

    for expert_index, expert in enumerate(eager_moe.experts):
        gemm_moe.gate_up_weight[expert_index, :, :gemm_moe.d_hidden,].copy_(
            expert.gate_proj.weight.T
        )
        gemm_moe.gate_up_weight[expert_index, :, gemm_moe.d_hidden:,].copy_(
            expert.a_up_proj.weight.T
        )
        gemm_moe.down_weight[expert_index].copy_(expert.a_down_proj.weight.T)

@torch.no_grad()
def load_eager_from_gemm(eager_moe: TopKSparseMoE, gemm_moe: GEMM_TopKSparseMoE) -> None:
    assert eager_moe.config.expert_num == gemm_moe.config.expert_num
    assert eager_moe.d_model == gemm_moe.d_model
    assert eager_moe.d_hidden == gemm_moe.d_hidden
    
    eager_moe.norm.load_state_dict(gemm_moe.norm.state_dict())
    eager_moe.router.load_state_dict(gemm_moe.router.state_dict())

    # [E, D, H], [E, D, H]
    gate_up_weight, a_up_proj = gemm_moe.gate_up_weight.split(
        [gemm_moe.d_hidden, gemm_moe.d_hidden],
        dim=-1,
    )
    for expert_index, expert in enumerate(eager_moe.experts):
        expert.gate_proj.weight.copy_(gate_up_weight[expert_index].T)
        expert.a_up_proj.weight.copy_(a_up_proj[expert_index].T)
        expert.a_down_proj.weight.copy_(gemm_moe.down_weight[expert_index].T)
    
def build_ffn(
    d_model: int,
    dropout: float,
    ffn_config: FFNConfig
) -> DenseFFN | TopKSparseMoE | GEMM_TopKSparseMoE:
    ffn_config.validate()
    if ffn_config.kind == "dense":
        ffn = DenseFFN(
            d_model=d_model,
            hidden_dim=ffn_config.hidden_dim,
            dropout=dropout,
        )
    else:
        assert ffn_config.moe is not None
        if ffn_config.backend == "eager":
            ffn = TopKSparseMoE(
                d_model,
                config=ffn_config.moe,
                hidden_dim=ffn_config.hidden_dim,
                dropout=dropout,
            )
        elif ffn_config.backend == "gemm":
            assert ffn_config.gemm is not None
            ffn = GEMM_TopKSparseMoE(
                d_model,
                config=ffn_config.moe,
                hidden_dim=ffn_config.hidden_dim,
                dropout=dropout,
                use_grouped_gemm=ffn_config.gemm.mode == "grouped",
            )
        else:
            raise ValueError(f"Unsupported FFN backend: {ffn_config.backend}")
    return ffn