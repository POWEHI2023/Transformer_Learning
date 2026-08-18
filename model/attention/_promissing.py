from __future__ import annotations

import torch
from dataclasses import dataclass

@ dataclass
class AttentionConfigPro:
    pass

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

class MultiHead_LatenAttention(torch.nn.Module):
    '''
    Laten Attention 通过 c = Wx，qk -> Wqc 优化 KV 存储量的同时不引入额外计算开销(每次kv=Wc)
    但是 RoPE 情况下 cache 中的 c 需要改为 c + k^{rope}, 因为 RoPE Attention Score 类似 qk + {q_r}{k_r}.
    MLA 只是通过低秩压缩使复杂度从 O(Ld) 中的 d 降低, 从 2 * n_head * d_head -> d_c, 后续引出 Linear Attention.

    KV Cache: c_kv[B, L, R_kv], k_rope[B, 1, L, D_r]
    '''
    def __init__(
        self,
        d_model: int,
        config: AttentionConfigPro,
    ) -> None:
        super().__init__()
        

class LinearAttention(torch.nn.Module):
    '''
    不再保存每个 token 的 KV, 普通 Softmax Attention: O = softmax(QK_T)V, 瓶颈在 QK_T,
    产生一个 LxL Attention Matrix, 复杂度为 O(L^2 x d), Decode 时单步计算复杂度 O(Ld), KV Cache 空间复杂度 O(Ld).
    Linear Attention: S_t = \sum [ f(k) x v^T ], O_t = f(q)^T x S_t, S_t = S_{t-1} + f(k) x v^T.
    利用 Kernel Feature Map 和矩阵结合率使得 Attention 变成一个 RNN, 计算复杂度随着序列长度线性扩展.
    Cache 空间复杂度: O(d^2), 且不随上下文增长, 历史 token 被压缩进 S_t.
    问题: 固定容量记忆, 历史信息都压缩进 S_t, 新旧信息叠加会产生 Memory Collision/ Interference/ Retrieval Degradation.
    -> 再次引出后续的 DeltaNet: 写新东西前先看看旧的记忆里都有什么.
    '''
    def __init__(
        self,
        d_model: int,
        config: AttentionConfigPro,
    ) -> None:
        super().__init__()


'''
至少在理论上补全:
DeltaNet -> Gated DeltaNet -> Kimi Delta Attention.
'''