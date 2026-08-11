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

    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,

    K: tl.constexpr,
    N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)

    problem_tile_start = 0
    row_start = 0

    for expert_id in range(num_experts):
        # 获得 token 的结束位置
        row_end = tl.load(offsets_ptr + expert_id)
        # 代表当前 expert 参与计算的 token 数量
        expert_m = row_end - row_start

        # 当前专家分到的需要计算的 token 有 expert_m 个, 需要分 num_m_tiles 次计算.
        num_m_tiles = tl.cdiv(expert_m, BLOCK_M)
        # 输出需要分 num_m_tiles 次计算.
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        # 输出矩阵共分成了 num_tiles 个小矩阵, 拼接后成为总的结果.
        num_tiles = num_m_tiles * num_n_tiles

        # 每次计算总的结果中的一个小块.
        # 当前专家输出 BLOCK 中包含的 tile 任务 id, 从 problem_tile_start 到 problem_tile_end.
        # 这个输出 BLOCK 代表了一个专家需要的完整计算.
        problem_tile_end = problem_tile_start + num_tiles

        '''
        假设当前需要计算的 token 数量为 6, 中间维度为 8, 输出维度为 6.
        BLOCK_M = 3, BLOCK_K = 4, BLOCK_N = 3.
        6 % BLOCK_M == 0, 8 % BLOCK_K == 0, 6 % BLOCK_N == 0.

        A(m0,k0) = A[0:3, 0:4]  # [3,4]
        A(m0,k1) = A[0:3, 4:8]  # [3,4]
        A(m1,k0) = A[3:6, 0:4]  # [3,4]
        A(m1,k1) = A[3:6, 4:8]  # [3,4]

        B(k0,n0) = B[0:4, 0:3]  # [4,3]
        B(k0,n1) = B[0:4, 3:6]  # [4,3]
        B(k1,n0) = B[4:8, 0:3]  # [4,3]
        B(k1,n1) = B[4:8, 3:6]  # [4,3]

        tile_m = 1, tile_n = 0, 表示取 A 矩阵的第一组 token; 取 B 矩阵的第 0 组 输出特征列.
        A 的第 1 组 为: [3:6, :], 分为连个 BLOCK_K 块, [3:6, 0:4], [3:6, 4:8].
        B 的第 0 组 为: [:, 0:3], [0:4, 0:3], [4:8, 0:3].

        根据 k_block 沿 K 维分成两次计算, [3:6, 0:4]x[0:4, 0:3], [3:6, 4:8]x[4:8, 0:3] -> [3:6, 0:3]
        '''
        while tile_id >= problem_tile_start and tile_id < problem_tile_end:
            # tile_id 减去当前专家包含的任务起始 id, 表示了当前专家中的第几个 tile 任务.
            local_tile_id = tile_id - problem_tile_start

            # 计算第几块中的 token 参与计算. 输出小块中的下标 m.
            tile_m = local_tile_id // num_n_tiles
            # 输出小块中的下标 n
            tile_n = local_tile_id % num_n_tiles

            # 左矩阵的 row_id.
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            # 右矩阵的 col_id.
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            # 在 FP32 中累加当前输出 tile 沿 K 维的部分乘积. [BLOCK_M, BLOCK_N]
            accumulator = tl.zeros(
                (BLOCK_M, BLOCK_N),
                dtype=tl.float32,
            )

            # 中间维度可以分成几次计算.
            for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                # 当前计算的中间维度.
                current_k = k_block * BLOCK_K + offs_k

                # 一个一维内存空间, 根据起始指针访问 第'expert_id'个专家的第'offs_m'些 token, 第'current_k'些中间维度.
                # a 输入的维度是 [R, K], 需要访问 [row_start + offs_m, current_k] 索引中的值.
                a_ptrs = (
                    a_ptr
                    + (row_start + offs_m[:, None]) * stride_am
                    + current_k[None, :] * stride_ak
                )

                # 同理, b 权重矩阵是 [E, K, N], 所以直接根据索引 [expert_id, current_k, offs_n] 访问内存空间.
                b_ptrs = (
                    b_ptr
                    + expert_id * stride_be
                    + current_k[:, None] * stride_bk
                    + offs_n[None, :] * stride_bn
                )

                # 从地址中加载数值, 并且 mask 掉无效值. 一般在边界才会有无效值.
                a = tl.load(a_ptrs, mask=((offs_m[:, None] < expert_m) & (current_k[None, :] < K)), other=0.0)
                b = tl.load(b_ptrs,mask=((current_k[:, None] < K) & (offs_n[None, :] < N)), other=0.0)

                # 累加结果.
                accumulator += tl.dot(a, b)

            # 存储结果的地址.
            c_ptrs = (
                c_ptr
                + (row_start + offs_m[:, None]) * stride_cm
                + offs_n[None, :] * stride_cn
            )

            '''
            mask 决定哪些地址真的从显存读取和哪些地址真的写入显存.
            它主要用来处理最后一个 tile 不足 BLOCK_M/BLOCK_N/BLOCK_K 的情况，避免越界访问.

            Load: mask=True  → 从对应地址读取/ mask=False → 不访问对应地址, 返回 other. other=0.0 不影响累加结果.
                [:, None] 约束行, [None, :] 约束列.
                行, offs_m = [64,65,...,127], 但当前 Expert 只有70行, offs_m < expert_m -> 64～69  → True/ 70～127 → False.
                列同理, 广播之后屏蔽行列中为 False 的值为 other.
            
            Store: mask=True  → 把 accumulator 写入 C/ mask=False → 不执行写入.
            '''
            # 存储结果.
            tl.store(c_ptrs, accumulator, mask=((offs_m[:, None] < expert_m) & (offs_n[None, :] < N)))

            # 循环执行下一个 Tile, NUM_SMS 表示启动的 persistent Triton program 数量.
            tile_id += NUM_SMS

        problem_tile_start = problem_tile_end
        row_start = row_end

'''
R = T * TopK
a: R 个 Token
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
        a, b, output,
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
    '''
    Tile 的含义: 把一个大矩阵乘法分成多个小矩阵乘法.
    完整的 GEMM 计算是: [M_e, K] @ [K, N] = [M_e, N], M_e 代表专家收到的 token 数量, K 是规约维度, N 是输出维度.
    把矩阵切成 BLOCK_M x BLOCK_N 的小块, 上述的 tile 大小为 [64, 64].

    BLOCK_M: 一个 tile 最多处理多少行, 处理多少个路由 token. 一个 tile 最多计算 64 个 token 的输出.
    BLOCK_N: 一个 tile 最多计算多少个输出特征.
    BLOCK_K: 一个 tile 每次处理多少个规约维度. 输出的 tile 始终是 [BLOCK_M, BLOCK_N], 但是为了计算它需要沿 K 维反复累加.
    '''

    return output

class GroupedMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offset):
        ctx.save_for_backward(a, b, offset)
        return group_mm_forward(a, b, offset)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, offsets = ctx.saved_tensors
        # dA_e = dC_e @ B_e.T
        # grad_output: [R,N]
        # B.transpose: [E,N,K]
        # offsets:     [E]

        # 上游产生的梯度可能是非连续的, transpose 或切片视图.
        grad_output = grad_output.contiguous()
        grad_a = None
        grad_b = None

        import torch.nn.functional as F
        if ctx.needs_input_grad[0]:
            grad_a = F.grouped_mm(
                grad_output,               # [R,N]
                b.transpose(-2, -1),       # [E,N,K]
                offs=offsets,
            )                              # [R,K]

        if ctx.needs_input_grad[1]:
            # dB_e = A_e.T @ dC_e
            grad_b = F.grouped_mm(
                a.transpose(0, 1),  # [K,R]
                grad_output,        # [R,N]
                offs=offsets,
            )                       # [E,K,N]

        return grad_a, grad_b, None

# dB_e = A_e.T @ dC_e
@triton.jit
def grouped_mm_bwd_weight_kernel(
    a_ptr,              # [R, K],
    grade_output_ptr,   # [R, N]
    grad_b_ptr,         # [E, K, N],
    offsets_ptr,        # [E],

    stride_ar, stride_ak,
    stride_or, stride_on,
    stride_be, stride_bk, stride_bn,

    K: tl.constexpr,    # [K, R] @ [R, N]
    N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_expert = num_k_tiles * num_n_tiles

    expert_id = pid // tiles_per_expert
    local_tile_id = pid % tiles_per_expert

    tile_k = local_tile_id // num_n_tiles
    tile_n = local_tile_id % num_n_tiles

    row_end = tl.load(offsets_ptr + expert_id)
    row_start = tl.load(
        offsets_ptr + expert_id - 1,
        mask=expert_id > 0,
        other=0,
    )
    expert_m = row_end - row_start

    offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros(
        (BLOCK_K, BLOCK_N),
        dtype=tl.float32,
    )

    for r_start in range(0, expert_m, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        # [R, K]
        _a = (
            a_ptr
            + (row_start + offs_r[:, None]) * stride_ar
            + offs_k[None, :] * stride_ak
        )
        a_block = tl.load(
            _a,
            mask=(
                (offs_r[:, None] < expert_m)
                & (offs_k[None, :] < K)
            ),
            other=0.0,
        )

        # [R, N]
        _grade = (
            grade_output_ptr
            + (row_start + offs_r[:, None]) * stride_or
            + offs_n[None, :] * stride_on
        )
        grade_block = tl.load(
            _grade,
            mask=(
                (offs_r[:, None] < row_end)
                & (offs_n[None, :] < N)
            ),
            other=0.0,
        )

        accumulator += tl.dot(a_block.T, grade_block)

    # grad_b_ptr: [E, K, N]
    _b = (
        grad_b_ptr
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    tl.store(
        _b,
        accumulator,
        mask=(
            (offs_k[:, None] < K) & (offs_n[None, :] < N)
        ),
    )

class GroupedMMFunction_Optim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offset) -> torch.Tensor:
        ctx.save_for_backward(a, b, offset)
        return group_mm_forward(a, b, offset)

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        '''TODO 后续通过一个 Triton Kernel 计算 dB'''
        a, b, offsets = ctx.saved_tensors
        # dA_e = dC_e @ B_e.T
        # dB_e = A_e.T @ dC_e -> 归约维度为可变维度 M_e: [K,M_e] @ [M_e,N] -> [K,N]

        # 上游产生的梯度可能是非连续的, transpose 或切片视图.
        grad_output = grad_output.contiguous()
        grad_a = None
        grad_b = None

        # 变长 M_e 不是归约维度, 在矩阵行方向, 复用现有的 forward kernel.
        # [R,K_in] @ [E,K_in,N_out] -> [R,N_out]
        import torch.nn.functional as F
        if ctx.needs_input_grad[0]:
            grad_a = group_mm_forward(
                grad_output,               # [R,N]
                b.transpose(-2, -1),       # [E,N,K]
                offsets,
            )                              # [R,K]

        # 归约维度变成了 M_e, 不能复用 group_mm_forward.
        if ctx.needs_input_grad[1]:
            grad_b = F.grouped_mm(
                a.transpose(0, 1),  # [K,R]
                grad_output,        # [R,N]
                offs=offsets,
            )                       # [E,K,N]

        return grad_a, grad_b, None