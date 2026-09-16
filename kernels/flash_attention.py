import math

import torch
import triton
import triton.language as tl

"""
参考 zflash_attention.py 中的实现, batch, heads, sequences 分别映射到 program grid
q_block_id = program_id(0)
bh_id = program_id(1)
batch_id = bh_id // H
head_id = bh_id % H
"""

@triton.jit
def flash_attention_v1_kernel(
    q_ptr, # [B, H, Q, D]
    k_ptr, # [B, H, K, D]
    v_ptr, # [B, H, K, D]
    o_ptr, # [B, H, Q, D]
    scale,
    q_start_position,
    k_start_position,

    q_valid_length, # [B]
    k_valid_length, # [B]

    stride_batch,
    stride_heads,
    stride_query,    
    stride_model,

    stride_key_batch,
    stride_key_heads,
    stride_key,
    stride_key_model,

    stride_q_length_batch,
    stride_k_length_batch,

    stride_o_batch,
    stride_o_heads,
    stride_o_query,
    stride_o_model,

    NUM_SMS,
    FULL_BATCH: tl.constexpr,
    FULL_HEADS: tl.constexpr,
    FULL_QUERY: tl.constexpr,
    FULL_KEY: tl.constexpr,
    FULL_MODEL: tl.constexpr,

    BLOCK_HEADS: tl.constexpr, # BLOCK_HEADS == FULL_HEADS
    BLOCK_QUERY: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_MODEL: tl.constexpr, # allow BLOCK_MODEL >= D
):
    # grid = (ceil_div(Q, BLOCK_QUERY), B * H)
    q_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    batch_id = bh_id // FULL_HEADS
    head_id = bh_id % FULL_HEADS

    # get local query block
    q_rows = q_block_id * BLOCK_QUERY + tl.arange(0, BLOCK_QUERY)
    q_cols = tl.arange(0, BLOCK_MODEL)

    q_ptrs = (
        q_ptr + stride_batch * batch_id + stride_heads * head_id
        + q_rows[:, None] * stride_query
        + q_cols[None, :] * stride_model
    )
    
    q_mask = (q_rows[:, None] < FULL_QUERY) & (q_cols[None, :] < FULL_MODEL)

    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    q_positions = q_start_position + q_rows
    q_length = tl.load(q_valid_length + stride_q_length_batch * batch_id)
    q_valid_mask = q_rows < q_length

    # get local key value block
    # k_base = k_ptr + stride_key_batch * batch_id + stride_key_heads * head_id
    # v_base = v_ptr + stride_key_batch * batch_id + stride_key_heads * head_id
    # kv_rows = tl.arange(0, FULL_KEY)
    # kv_cols = tl.arange(0, BLOCK_MODEL)

    # k_ptrs = (
    #     k_base
    #     + kv_rows[:, None] * stride_key
    #     + kv_cols[None, :] * stride_key_model
    # )
    # v_ptrs = (
    #     v_base
    #     + kv_rows[:, None] * stride_key
    #     + kv_cols[None, :] * stride_key_model
    # )

    # kv_mask = (kv_rows[:, None] < FULL_KEY) & (kv_cols[None, :] < FULL_MODEL)

    # k = tl.load(k_ptrs)
    # v = tl.load(v_ptrs)
    #
    # ->> 直接在循环里按 BLOCK_KEY 读取 Key Value.
    # ---------- ---------- ---------- ---------- ----------

    # Online Softmax loop
    m_global = tl.full((BLOCK_QUERY,), -float("inf"), tl.float32)
    l_global = tl.zeros((BLOCK_QUERY,), tl.float32)
    # [Q, D] @ [D, K] @ [K, D] = [Q, D]
    o_global = tl.zeros((BLOCK_QUERY, BLOCK_MODEL), tl.float32)

    k_base = k_ptr + stride_key_batch * batch_id + stride_key_heads * head_id
    v_base = v_ptr + stride_key_batch * batch_id + stride_key_heads * head_id

    for kv_start in range(0, FULL_KEY, BLOCK_KEY):
        kv_rows = kv_start + tl.arange(0, BLOCK_KEY)
        kv_cols = tl.arange(0, BLOCK_MODEL)

        k_ptrs = (
            k_base
            + kv_rows[:, None] * stride_key
            + kv_cols[None, :] * stride_key_model
        )
        v_ptrs = (
            v_base
            + kv_rows[:, None] * stride_key
            + kv_cols[None, :] * stride_key_model
        )

        kv_mask = (kv_rows[:, None] < FULL_KEY) & (kv_cols[None, :] < FULL_MODEL)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        k_positions = k_start_position + kv_rows
        k_length = tl.load(k_valid_length + stride_k_length_batch * batch_id)
        k_valid_mask = kv_rows < k_length

        # [Q_BLOCK, K_BLOCK]
        s = tl.dot(q, tl.trans(k)) * scale

        # 这里的 Mask 使用是正确的吗？有没有缺了什么条件？
        causal_mask = q_positions[:, None] < k_positions[None, :]
        valid_mask = q_valid_mask[:, None] & k_valid_mask[None, :]
        blocked_mask = (
            ~valid_mask | causal_mask
            | (q_rows >= FULL_QUERY)[:, None]
            | (kv_rows >= FULL_KEY)[None, :]
        )

        s = tl.where(blocked_mask, -float("inf"), s)

        m_local = tl.maximum(m_global, tl.max(s, axis=1)) # [BLOCK_QUERY]
        safe_m_local = tl.where(m_local == -float("inf"), 0.0, m_local)

        p = tl.exp(s - safe_m_local[:, None]) # e ^ (s - new global m)
        p = tl.where(blocked_mask, 0.0, p)

        alpha = tl.exp(m_global - safe_m_local) # e ^ (global m - new global m)
        l_global = l_global * alpha + tl.sum(p, axis=1)

        o_local = tl.dot(p.to(v.dtype), v)
        o_global = o_global * alpha[:, None] + o_local
        m_global = m_local

    # Q @ K^t x V
    saft_l = tl.where(l_global > 0.0, l_global, 1.0)
    output = o_global / saft_l[:, None]
    output = tl.where((l_global > 0.0)[:, None], output, 0.0)

    o_ptrs = (
        o_ptr
        + batch_id * stride_o_batch
        + head_id * stride_o_heads
        + q_rows[:, None] * stride_o_query
        + q_cols[None, :] * stride_o_model
    )
    o_mask = (q_rows[:, None] < FULL_QUERY) & (q_cols[None, :] < FULL_MODEL)
    tl.store(o_ptrs, output, mask=o_mask)


