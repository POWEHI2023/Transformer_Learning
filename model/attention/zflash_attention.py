from __future__ import annotations

import torch
import math


# Multi Head Attention -> Flash Attention Backend
def flash_attention_forward(
    q: torch.Tensor, # [B, H, QO, D]
    k: torch.Tensor, # [B, H, KV, D]
    v: torch.Tensor, # [B, H, KV, D]
    *,
    block_q: int = 64,
    block_kv: int = 64,
    causal: bool = True,
    key_padding_mask: torch.Tensor | None = None, # [B, K], True means valid
    query_padding_mask: torch.Tensor | None = None, # [B, Q], True means valid
    query_start_position: int = 0,
) -> torch.Tensor: # output: [B, H, QO, D]
    output_blocks = []

    # force q shape dimension is 4
    batch_size, n_heads, query_length, head_dim = q.shape
    _, _, key_value_length, _ = q.shape

    for q_start in range(0, query_length, block_q):
        q_end = min(q_start + block_q, query_length)

        # [B, H, M, D]
        q_block = q[:, :, q_start:q_end, :]
        _, _, current_q_length, _ = q_block.shape

        negative_large = torch.finfo(torch.float32).min

        # 扫描 KV 范围内每个 Q 行的最大 Score/ global m
        row_max = torch.full(
            (batch_size, n_heads, current_q_length),
            negative_large,
            dtype=torch.float32,
            device=q.device,
        )
        # 扫描范围内的分母/ global l
        row_sum = torch.zeros_like(row_max)
        # 未除以 softmax 分母的输出, A = sum(e^(s - m))
        accumulator = torch.zeros(
            (batch_size, n_heads, current_q_length, head_dim),
            dtype=torch.float32,
            device=q.device,
        )

        for kv_start in range(0, key_value_length, block_kv):
            kv_end = min(kv_start + block_kv, key_value_length)

            # [B, H, N, D]
            k_block = k[:, :, kv_start:kv_end, :]
            v_block = v[:, :, kv_start:kv_end, :]

            scores = torch.matmul(
                q_block.float(),
                k_block.float().transpose(-2, -1),
            )
            scores *= 1.0 / math.sqrt(head_dim) # [B, H, M, N]

            query_positions = (
                query_start_position
                + torch.arange(q_start, q_end, device=q.device)
            )
            key_positions = torch.arange(
                kv_start, kv_end, device=q.device
            )
            if causal:
                blocked = ( # [1, 1, M, N]
                    key_positions[None, None, None, :]
                    > query_positions[None, None, :, None]
                )
            else:
                blocked = torch.zeros(
                    (1, 1, current_q_length, kv_end - kv_start),
                    dtype=torch.bool,
                    device=q.device,
                )
            current_valid_keys = key_padding_mask[:, kv_start:kv_end]
            blocked = (
                blocked | ~current_valid_keys[:, None, None, :]
            )
            current_valid_query = query_padding_mask[:, q_start:q_end]
            blocked = (
                blocked | ~current_valid_query[:, None, :, None]
            )
            masked_socres = scores.masked_fill(blocked, negative_large)


            # ---------- m, l, A, Online Softmax 开始咯 ----------

            # local m
            block_max = masked_socres.amax(dim=-1)
            # new global m = max(global m, local m)
            new_row_max = torch.maximum(row_max, block_max)
            
            # e^(global m - new global m)
            old_scale = torch.exp(row_max - new_row_max)

            # local l = e^(local s - new global max)
            probailities = torch.exp(masked_socres - new_row_max.unsqueeze(-1))
            probailities = probailities.masked_fill(blocked, 0.0)

            # global l * e^(m - M) + local l = new blobal l
            new_row_sum = (
                row_sum * old_scale + probailities.sum(dim=-1)
            )

            # local A = local l * local v
            # new global A = global A * e^(global m - new m) + local A
            accumulator = (
                accumulator * old_scale.unsqueeze(-1)
                + torch.matmul(probailities, v_block.float())
            )

            # new global m
            row_max = new_row_max
            # new global l
            row_sum = new_row_sum

        # global l
        safe_denominator = row_sum.clamp_min(1e-20)
        # output = 1/l * A
        output_block = (
            accumulator / safe_denominator.unsqueeze(-1)
        )


        # - - - - | | | |   - - - -   - - - -
        # - - - - | | | |   - - - -   - - - -
        # | | | | | | | |   - - - -   | | | |
        # | | | | | | | |   - - - -   | | | |
        #                   | | | |
        #                   | | | |
        #                   | | | |
        #                   | | | |
        #
        # 内层循环的 QK @ V -> O 示意图
        # 
        # - - - -   - |   - |
        # | | | |   - |   | |
        #           - |
        #           - |
        # 上面的 QK 只取了左上角, 外层获得 QK 的示意图

        output_block = torch.where(
            row_sum.unsqueeze(-1) > 0,
            output_block,
            torch.zeros_like(output_block),
        )

        output_blocks.append(output_block.to(q.dtype))

    return torch.cat(output_blocks, dim=2)
    

