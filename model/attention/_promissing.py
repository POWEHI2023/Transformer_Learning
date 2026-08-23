from __future__ import annotations

import torch
from configs.zconfig import AttentionConfig
from model.position.zposition import RoPE
from model.attention.zattention import AttentionOutput
from dataclasses import dataclass


@dataclass
class MLAConfig:
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    value_head_dim: int

    def __post_init__(self) -> None:
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("")

class MultiHeadLatentAttention(torch.nn.Module):
    '''
    Laten Attention 通过 c = Wx，qk -> Wqc 优化 KV 存储量的同时不引入额外计算开销(每次kv=Wc)
    但是 RoPE 情况下 cache 中的 c 需要改为 c + k^{rope}, 因为 RoPE Attention Score 类似 qk + {q_r}{k_r}.
    MLA 只是通过低秩压缩使复杂度从 O(Ld) 中的 d 降低, 从 2 * n_head * d_head -> d_c, 后续引出 Linear Attention.

    KV Cache: c_kv[B, L, R_kv], k_rope[B, 1, L, D_r]
    '''
    def __init__(
        self,
        d_model: int,
        attention_config: AttentionConfig,
        mla_config: MLAConfig,
    ) -> None:
        super().__init__()

        if attention_config.kind != "mla":
            raise ValueError(f"Expected attention kind 'mla', got {attention_config.kind}")
        if attention_config.backend != "eager":
            raise ValueError("Initial MLA implementation only supports eager backend")

        self.d_model = d_model   
        self.n_heads = attention_config.n_heads # Query Head 的数量, MLA 输出的 Attention Head 的数量
        self.use_causal_mask = attention_config.use_causal_mask

        # Query 的低秩压缩维度: [B, L, D] -> [B, L, Rq], 后面通过 up_proj -> [B, L, H * (Dc + Dr)]
        self.q_lora_rank = mla_config.q_lora_rank
        # KV 的联合低秩压缩维度, [B, L, D] -> [B, L, Rkv]
        # down_proj: [B, L, D] -> [B, L, Rkv + Dr] (kv_lora_rank & qk_nope_head_dim)
        self.kv_lora_rank = mla_config.kv_lora_rank

        # 每个 Attention Head 中不使用 RoPE 的内容维度. QK 用来计算的维度. 不包含位置维度.
        # q_nope^T · k_ope -> [B, H, Dc, L] · [B, H, L, Dc] -> [B, H, Dc, Dc]
        # 先得到吸收权重的 Q, 然后再与 kv_latent 计算内容分数. 
        # q_absorbed: [B, H, Q, Rkv] & kv_latent: [B, K, Rkv]
        self.qk_nope_head_dim = mla_config.qk_nope_head_dim

        # 每个 Head 中专门负责位置信息的维度: q_rope: [B, H, L, Dr], k_rope: [B, 1, L, Dr]
        self.qk_rope_head_dim = mla_config.qk_rope_head_dim
        # 从 latent 映射回 value 时的 value_head_dim. 
        # [B, L, Rkv] -> [B, L, H * (K + V)]
        self.value_head_dim = mla_config.value_head_dim

        # Query 中的 head_dim. [B, L, D] -> [B, L, Rq] -> [B, L, H * Dq] == [B, L, H * (Dc + Dr)]
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # x -> Query latent
        self.q_down_proj = torch.nn.Linear(d_model, self.q_lora_rank, bias=False)
        self.q_norm = torch.nn.RMSNorm(self.q_lora_rank)
        # Query latent -> 每个 head 的 q_nope 和 q_rope
        self.q_up_proj = torch.nn.Linear(
            self.q_lora_rank,
            self.n_heads * self.q_head_dim,
            bias=False,
        )

        # x -> KV latent + 共享的 RoPE Key
        self.kv_down_proj = torch.nn.Linear(
            d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = torch.nn.RMSNorm(self.kv_lora_rank)
        # KV latent -> 每个 head 的 k_nope 和 value
        # 原先是 Q @ KlW == Q @ K, 但是 KlW 计算消耗太大, 等价计算 WQ @ Kl.
        # Q 的 qk_nope_head_dim -> kv_lora_rank 然后和 K 的 kv_lora_rank 计算.
        self.kv_up_proj = torch.nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (
                self.qk_nope_head_dim
                + self.value_head_dim
            ),
            bias=False,
        )

        self.output_proj = torch.nn.Linear(
            self.n_heads * self.value_head_dim,
            d_model,
            bias=False,
        )
        self.rope = RoPE(
            head_dim=self.qk_rope_head_dim,
            base=attention_config.rope_base,
        )
        self.scale = self.q_head_dim ** -0.5

    def _project_query(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''返回 q 的 latent value 和 q 的位置编码'''
        batch_size, sequence_length, _ = x.shape

        # [B, L, D] -> [B, L, Rq]
        q_latent = self.q_down_proj(x)
        q_latent = self.q_norm(q_latent)

        # [B, L, Rq] -> [B, L, H * (Dc + Dr)]
        q = self.q_up_proj(q_latent)
        # [B, H, L, (Dc + Dr)]
        q = q.view(batch_size, sequence_length, self.n_heads, self.q_head_dim).transpose(1, 2)

        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = self.rope(q_rope, position_ids)

        return q_nope, q_rope

    def _project_kv(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''返回 kv 的 latent value 和 k 的位置编码 '''
        # [B, L, D] -> [B, L, Rkv + Dr] (kv_lora_dim + k_rope_dim)
        kv_and_rope = self.kv_down_proj(x)
        # [B, L, Rkv], [B, L, Dr]
        kv_latent, k_rope = kv_and_rope.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)

        # [B, L, Dr] -> [B, 1, L, Dr]
        k_rope = k_rope.unsqueeze(1)
        k_rope = self.rope(k_rope, position_ids) # [B, 1, L, Dr]

        return kv_latent, k_rope

    def _resolve_cache(
        self,
        kv_latent: torch.Tensor,
        k_rope: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_size, sequence_length, kv_lora_rank = kv_latent.shape
        if past_key_value is None:
            return kv_latent, k_rope, 0

        # [B, Lh, Rkv], [B, 1, Lh, Dr]
        cached_kv_latent, cached_k_rope = past_key_value

        if cached_kv_latent.ndim != 3:
            raise ValueError("MLA cached latent must have shape [B,K,Rkv]")
        if cached_k_rope.ndim != 4:
            raise ValueError("MLA cached RoPE key must have shape [B,1,K,Dr]")

        if cached_kv_latent.shape[0] != batch_size:
            raise ValueError("MLA cache batch size mismatch")
        if cached_kv_latent.shape[-1] != self.kv_lora_rank:
            raise ValueError("MLA cache latent dimension mismatch")

        if cached_k_rope.shape[0] != batch_size:
            raise ValueError("MLA RoPE cache batch size mismatch")
        if cached_k_rope.shape[1] != 1:
            raise ValueError("MLA RoPE cache must have one shared head")
        if cached_k_rope.shape[-1] != self.qk_rope_head_dim:
            raise ValueError("MLA RoPE cache dimension mismatch")

        past_length = cached_kv_latent.shape[1]
        if cached_k_rope.shape[2] != past_length:
            raise ValueError("MLA cache sequence lengths do not match")

        kv_latent = torch.cat([cached_kv_latent, kv_latent], dim=1)
        k_rope = torch.cat([cached_k_rope, k_rope], dim=2)

        return kv_latent, k_rope, past_length

    def _build_blocked_mask(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        past_length: int,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        key_length = past_length + sequence_length
        blocked_mask = None

        if self.use_causal_mask:
            query_positions = past_length + torch.arange(sequence_length, device=device)
            key_positions = torch.arange(key_length, device=device)

            # [1,1,Q,K]
            blocked_mask = (
                key_positions[None, :] > query_positions[:, None]
            )[None, None, :, :]

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, key_length):
                raise ValueError(
                    "With MLA cache, attention_mask must cover "
                    f"all keys and have shape {(batch_size, key_length)}"
                )

            valid_keys = attention_mask.to(device=device, dtype=torch.bool)

            if not valid_keys.any(dim=-1).all():
                raise ValueError("Every sample must contain at least one valid key")

            padding_blocked = ~valid_keys[:, None, None, :]
            blocked_mask = (
                padding_blocked
                if blocked_mask is None
                else blocked_mask | padding_blocked
            )

        return blocked_mask

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        if x.ndim != 3:
            raise ValueError("MLA input must have shape [B,Q,D]")
        batch_size, sequence_length, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")
        if past_key_value is not None and not use_cache:
            raise ValueError("past_key_value requires use_cache=True")

        q_nope, q_rope = self._project_query(x, position_ids)
        kv_latent, k_rope = self._project_kv(x, position_ids)
        kv_latent, k_rope, past_length = self._resolve_cache(kv_latent, k_rope, past_key_value)

        blocked_mask = self._build_blocked_mask(
            batch_size=batch_size,
            sequence_length=sequence_length,
            past_length=past_length,
            attention_mask=attention_mask,
            device=x.device,
        )

        kv_up_weight = self.kv_up_proj.weight.view(
            self.n_heads,
            self.qk_nope_head_dim + self.value_head_dim,
            self.kv_lora_rank, # input dim
        )
        # [n_heads, qk_nope_head_dim, kv_lora_rank], [n_heads, value_head_dim, kv_lora_rank]
        k_up_weight, v_up_weight = kv_up_weight.split(
            [self.qk_nope_head_dim, self.value_head_dim],
            dim=1,
        )

        # [B, H, L, qk_nope_head_dim] x [H, qk_nope_head_dim, kv_lora_rank]
        # -> [B, H, L, kv_lora_rank]
        q_absorbed = torch.einsum(
            "bhqd,hdr->bhqr",
            q_nope,
            k_up_weight,
        )
        # [B, H, L, kv_lora_rank] @ [B, Lk, kv_lora_rank]
        # -> [B, H, L, Lk]
        content_scores = torch.einsum(
            "bhqr,bkr->bhqk",
            q_absorbed,
            kv_latent,
        )

        # decoupled RoPE
        position_scores = q_rope @ k_rope.transpose(-2, -1)

        scores = (content_scores + position_scores) * self.scale
        if blocked_mask is not None:
            scores = scores.masked_fill(blocked_mask, float("-inf"))
        attention_weights = torch.softmax(scores.float(), dim=-1).to(dtype=kv_latent.dtype)

        # S @ V: [B, H, L, Lk] @ [B, Lk, kv_lora_rank] -> [B, H, L, kv_lora_rank]
        latent_output = torch.einsum(
            "bhqk,bkr->bhqr",
            attention_weights,
            kv_latent,
        )

        # [B, H, L, kv_lora_rank] @ [H, value_head_dim, kv_lora_rank]
        # -> [B, H, L, value_head_dim]
        head_output = torch.einsum(
            "bhqr,hvr->bhqv",
            latent_output,
            v_up_weight,
        )

        # [B, H, L, value_head_dim] -> [B, L, H * value_head_dim]
        output = head_output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.n_heads * self.value_head_dim,
        )

        return AttentionOutput(
            output=self.output_proj(output),
            cached_key_value=(kv_latent, k_rope) if use_cache else None,
        )


@dataclass
class LinearAttentionConfig:
    pass

class LinearAttention(torch.nn.Module):
    '''
    不再保存每个 token 的 KV, 普通 Softmax Attention: O = softmax(QK_T)V, 瓶颈在 QK_T,
    产生一个 LxL Attention Matrix, 复杂度为 O(L^2 x d), Decode 时单步计算复杂度 O(Ld), KV Cache 空间复杂度 O(Ld).
    Linear Attention: S_t = sum [ f(k) x v^T ], O_t = f(q)^T x S_t, S_t = S_{t-1} + f(k) x v^T.
    利用 Kernel Feature Map 和矩阵结合率使得 Attention 变成一个 RNN, 计算复杂度随着序列长度线性扩展.
    Cache 空间复杂度: O(d^2), 且不随上下文增长, 历史 token 被压缩进 S_t.
    问题: 固定容量记忆, 历史信息都压缩进 S_t, 新旧信息叠加会产生 Memory Collision/ Interference/ Retrieval Degradation.
    -> 再次引出后续的 DeltaNet: 写新东西前先看看旧的记忆里都有什么.
    '''
    def __init__(
        self,
        d_model: int,
        attention_config: AttentionConfig,
        config: LinearAttentionConfig,
    ) -> None:
        super().__init__()


'''
至少在理论上补全:
DeltaNet -> Gated DeltaNet -> Kimi Delta Attention.
'''