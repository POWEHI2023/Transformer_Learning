import torch
import triton
import triton.language as tl

@triton.jit
def grouped_mm_fwd_kernel(
    a_ptr,          # [R, K]
    b_ptr,          # [E, K, N]
    c_ptr,          # [R, N]
    offsets_ptr,    # [E]

    num_experts,

    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,

    K: tl.constexpr,
    N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 每个 program 相当于一个持久运行的 CTA
    tile_id = tl.program_id(0)

    problem_tile_start = 0
    row_start = 0

    for expert_id in range(num_experts):
        row_end = tl.load(offsets_ptr + expert_id)
        expert_m = row_end - row_start

        num_m_tiles = tl.cdiv(expert_m, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        problem_tile_end = problem_tile_start + num_tiles

        # 当前 CTA 可能连续处理多个 tile
        while tile_id >= problem_tile_start and tile_id < problem_tile_end:
            local_tile_id = tile_id - problem_tile_start

            tile_m = local_tile_id // num_n_tiles
            tile_n = local_tile_id % num_n_tiles

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            accumulator = tl.zeros(
                (BLOCK_M, BLOCK_N),
                dtype=tl.float32,
            )

            for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                current_k = k_block * BLOCK_K + offs_k

                a_ptrs = (
                    a_ptr
                    + (row_start + offs_m[:, None]) * stride_am
                    + current_k[None, :] * stride_ak
                )

                b_ptrs = (
                    b_ptr
                    + expert_id * stride_be
                    + current_k[:, None] * stride_bk
                    + offs_n[None, :] * stride_bn
                )

                a = tl.load(
                    a_ptrs,
                    mask=(
                        (offs_m[:, None] < expert_m)
                        & (current_k[None, :] < K)
                    ),
                    other=0.0,
                )

                b = tl.load(
                    b_ptrs,
                    mask=(
                        (current_k[:, None] < K)
                        & (offs_n[None, :] < N)
                    ),
                    other=0.0,
                )

                accumulator += tl.dot(a, b)

            c_ptrs = (
                c_ptr
                + (row_start + offs_m[:, None]) * stride_cm
                + offs_n[None, :] * stride_cn
            )

            tl.store(
                c_ptrs,
                accumulator,
                mask=(
                    (offs_m[:, None] < expert_m)
                    & (offs_n[None, :] < N)
                ),
            )

            tile_id += NUM_SMS

        problem_tile_start = problem_tile_end
        row_start = row_end

'''
R = T * TopK
a: T * K 个 Token
b: 所有 Expert 的权重打包之后的三维张量 [E, D, N]
D: 输入维度, N: 输入乘以权重之后的输出维度
[M_e, D] @ [D, N] -> [M_e, N] -> Expert 中包含 M_e 个 Token

offsets: 保存每个 Expert 在 a 中的累计结束位置
'''
def group_mm_forward(
    a: torch.Tensor,        # [R, D]
    b: torch.Tensor,        # [E, D, N]
    offsets: torch.Tensor,  # [E]
) -> torch.Tensor:          # [R, N]
    if not a.is_cuda or not b.is_cuda or not offsets.is_cuda:
        raise ValueError("Triton Group GEMM requires CUDA tensors.")
    if a.ndim != 2 or b.ndim != 3 or offsets.ndim != 1:
        raise ValueError("Excepted A=[R, K], B=[E, K, N], offsets=[E]")

    route_num, k = a.shape
    expert_num, weight_k, n = b.shape

    if weight_k != k:
        raise ValueError("A and B reduction dimensions do not match.")
    if offsets.shape[0] != expert_num:
        raise ValueError("offsets length must equal expert count.")
    if offsets.dtype != torch.int32:
        raise ValueError("offsets must use torch.int32")

    # 预先分配输出空间, 输出维度为: [R, N]
    output = torch.empty((route_num, n), dtype=a.dtype, device=a.device)
    # 获取 SM 的数量, 意味着启动多少个 persistent Triton programs.
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    grouped_mm_fwd_kernel[(num_sms,)](
        a,
        b,
        output,
        offsets,
        expert_num,

        a.stride(0),    #  a 在 R 维度移动需要跨过多少元素
        a.stride(1),    #  a 在 D 维度移动需要跨过多少元素
        b.stride(0),    #  b 在 E 维度移动需要跨过多少元素
        b.stride(1),    #  b 在 D 维度移动需要跨过多少元素
        b.stride(2),    #  b 在 N 维度移动需要跨过多少元素
        output.stride(0),   #  c 在 R 维度移动需要跨过多少元素
        output.stride(1),   #  c 在 N 维度移动需要跨过多少元素

        K=k,    # D 维度这里叫做 K, 规约维度
        N=n,
        NUM_SMS=num_sms,
        BLOCK_M=64, # 设置 tile 大小
        BLOCK_N=64, # 设置 tile 大小
        BLOCK_K=32, # 设置 tile 大小
        num_warps=4,
        num_stages=3,
    )

    return output