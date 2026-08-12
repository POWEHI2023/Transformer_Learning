# Transformer 多维并行重构计划

## 1. 文档目的

本文档面向当前仓库中的 Decoder-only Transformer，规划从现有单进程完整模型逐步重构到支持以下并行能力的训练系统：

- DP：Data Parallelism。
- TP：Tensor Parallelism。
- SP：Sequence Parallelism。
- PP/VPP：Pipeline Parallelism / Virtual Pipeline Parallelism。
- EP/ETP：Expert Parallelism / Expert Tensor Parallelism。
- CP：Context Parallelism。
- Distributed Optimizer 和分布式 checkpoint。

本计划采用 Megatron-Core 作为最终并行运行时。现有 Attention、Dense FFN、Router、MoE 和 Grouped GEMM 实现继续作为单卡参考实现，用于理解算法和进行数值对齐；不在生产训练路径中重复实现 Megatron-Core 已经提供的通信调度。

本文档替代 `archive/todoplan.md` 中已有的 TP/DP/EP 初步规划，但不删除旧文档。

## 2. 当前状态与主要缺口

### 2.1 当前模型

- `model/ztransformer.py`
  - 每个进程构造完整 Embedding、全部 TransformerBlock、FinalNorm 和 LM Head。
  - 内部激活布局为 `[B, S, H]`。
  - Embedding 和 LM Head 可以绑定权重。
  - 所有层的 MoE auxiliary loss 在完整模型末尾统一汇总。
- `model/zattention.py`
  - QKV 和输出投影均为普通 `torch.nn.Linear`。
  - 每个进程持有全部 Attention heads 和完整 sequence。
  - causal mask 通过本地 `arange` 构造，尚不能表示 CP 下的全局 query/key 位置。
  - MHA、MQA、GQA 的实现路径不同；GQA 已支持 eager/einsum，但没有分片 head 语义。
- `model/zmoe.py`
  - Router 输出全局 expert id。
  - 每张设备保存全部 experts。
  - Token dispatch 只有本地排序，没有跨 rank All-to-All。
  - Grouped GEMM 权重第一维为全部 experts。
- `train.py`
  - 单进程 DataLoader、普通 AdamW、普通 Cross Entropy 和直接 `loss.backward()`。
  - 没有 distributed initialization、microbatch pipeline schedule 或进程组级指标规约。
- `model/mstore.py`
  - checkpoint 保存完整 model/optimizer state，不能描述 TP/PP/EP shard。
- `configs/zconfig.py`
  - `ParallelConfig` 仍是占位结构，缺少 CP、ETP、VPP、通信重叠和 distributed optimizer 配置。

## 3. 总体技术决策

### 3.1 使用 Megatron-Core 的范围

直接复用以下能力：

- `parallel_state` / `ProcessGroupCollection`：构造和传递并行进程组。
- `ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding`。
- Sequence Parallel 的 All-Gather / Reduce-Scatter autograd mappings。
- Vocab-parallel cross entropy。
- Pipeline P2P communicator 和 1F1B/interleaved 1F1B schedules。
- Megatron DDP、gradient finalize 和 Distributed Optimizer。
- MCore MoE Router、Token Dispatcher、Grouped GEMM、EP/ETP group。
- Transformer Engine Attention 和 Context Parallel 通信。
- Distributed checkpoint。

仓库负责以下部分：

- 模型配置到 MCore 配置的适配。
- 自定义 Transformer stage 的组装。
- 数据、loss、日志、checkpoint 入口和训练生命周期。
- 单卡参考实现及其与分布式实现的等价性测试。

### 3.2 Dense 网格和 Expert 网格分开建模

不要把所有维度机械地相乘为一个固定公式。采用 MCore parallel folding 的两套视图：

```text
Dense/Attention: world_size = TP × CP × DP × PP
MoE Experts:     world_size = ETP × EP × EDP × PP
```

- SP 复用 TP group，不增加 world size。
- VPP 是 PP rank 上的虚拟 chunk，不增加 world size。
- EP/ETP 和 dense TP/CP 可以采用不同的映射。
- 所有 group 的 rank 映射由统一运行时生成，模型模块不得自行推导 global rank。

### 3.3 内部张量布局

外部 batch 保持：

```text
token_ids / labels / attention_mask: [B, S]
```

分布式模型内部统一迁移到：

```text
hidden_states: [S_local, B, H]
```

原因：

- 与 MCore pipeline tensor shape 和 Sequence Parallel 的默认语义一致。
- SP/CP 都沿第一维切分，减少重复转置和 dim 参数分歧。
- Embedding 后只进行一次 `[B,S,H] -> [S,B,H]` 转换。

单卡参考模型可以暂时保留 `[B,S,H]`，测试适配器负责布局转换。最终应考虑让参考和分布式实现共享明确的 layout helper。

### 3.4 配置与模型结构解耦

- `ModelConfig` 只描述模型语义，不能因为 TP/PP 数量变化而改变 checkpoint 中的模型定义。
- `ParallelConfig` 只描述运行拓扑和优化开关。
- checkpoint 必须能够在兼容条件下使用不同并行拓扑重新加载。

### 3.5 正确性优先于通信重叠

每个并行维度首先完成同步、易验证版本，然后再启用：

- TP communication overlap。
- DP gradient reduce overlap。
- Distributed Optimizer parameter gather overlap。
- PP P2P overlap。
- EP All-to-All overlap。
- CP ring attention overlap。

## 4. 目标代码结构

计划逐步形成以下目录结构：

```text
configs/
  zconfig.py
  zparser.py

model/
  ztransformer.py             # 单卡参考模型/兼容入口
  zattention.py               # 单卡参考 Attention
  zmoe.py                     # 单卡参考 MoE
  distributed_transformer.py  # stage-local 分布式模型
  mcore_attention.py           # MCore/TE Attention 适配
  mcore_ffn.py                 # Dense TP FFN 和 MoE 适配

parallel/
  runtime.py                  # torch.distributed + MCore group 初始化
  config.py                   # 并行配置校验和派生值
  batch.py                    # DP 数据选择、CP/SP batch 切分
  losses.py                   # vocab-parallel/token-normalized loss
  grad.py                     # 全局 grad norm、gradient finalize

training/
  forward_step.py             # 提供给 pipeline schedule 的 forward_step_func
  trainer.py                  # microbatch、schedule、optimizer 生命周期
  metrics.py                  # 指标在正确 group 上规约
  checkpoint.py               # distributed checkpoint 适配

tests/
  unit/
  distributed/
  integration/
```

目录在对应阶段实际需要时创建，避免一次性建立空文件。

## 5. 配置设计

### 5.1 ParallelConfig 目标字段

```python
@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    virtual_pipeline_parallel_size: int | None = None
    context_parallel_size: int = 1

    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    sequence_parallel: bool = False
    cp_comm_type: str = "p2p"
    rank_order: str = "tp-cp-ep-dp-pp"

    use_distributed_optimizer: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    tp_comm_overlap: bool = False
    overlap_pp_p2p: bool = False
    overlap_ep_comm: bool = False

    backend: str = "nccl"
    timeout_minutes: int = 30
```

### 5.2 初始化前统一校验

所有 rank 在创建 process group 前执行相同的纯函数校验：

- 所有 parallel size 必须为正整数。
- `world_size` 满足 dense grid 和 expert grid 约束。
- `d_model % TP == 0`。
- `n_heads % TP == 0`。
- `hidden_dim % TP == 0`。
- `expert_num % EP == 0`。
- `expert_hidden_dim % ETP == 0`。
- TP 与 EP 同时大于 1 时必须启用 SP。
- VPP 启用时 PP 必须大于 1。
- 第一版 PP 要求 layer 数可以均匀分配；自定义不均匀 layout 在后续扩展。
- CP 第一版要求 sequence length 可被 CP 整除。
- SP 启用时，进入 sequence-sharded 区域的 sequence length 可被 TP 整除。
- vocab size 使用 padded vocab size 对齐 TP shard。

GQA 第一版增加：

- `n_kv_heads % TP == 0`。

MQA 的 KV replicated TP 路径单独列为后续任务，不允许静默使用错误切分。

## 6. 里程碑 M0：冻结单卡参考基线

### 6.1 工作项

- 为 Dense 和 MoE 模型增加固定 seed 的最小 golden 配置。
- 固定以下参考数据：
  - 无 Padding batch。
  - 右侧 Padding batch。
  - MoE 极不均匀路由 batch。
- 保存或动态生成参考：
  - logits。
  - LM loss、aux loss、z loss。
  - input gradient。
  - QKV、output projection、FFN、Router、Expert 的参数梯度。
  - 一次 AdamW step 后参数。
- 测试中关闭 Dropout；随机性相关测试独立处理。
- 修复现有 `load_model_config()` 和 `model_config_from_dict()` 未完成实现，保证 YAML 与 CLI 使用同一校验路径。

### 6.2 验收门槛

- CPU 上 Dense/MoE 单步训练测试稳定通过。
- checkpoint 保存、加载和 resume 结果一致。
- 后续任一并行实现均可使用本阶段的完整模型作为数值 oracle。

## 7. 里程碑 M1：分布式运行时和 DP 基线

### 7.1 运行时初始化

- 读取 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`。
- 设置当前 CUDA device 后初始化默认 process group。
- 调用 MCore `initialize_model_parallel()` 创建 TP、PP、DP、CP、EP 等 group。
- 使用 `ProcessGroupCollection` 显式传递 group，避免模型深处依赖隐式 global state。
- 增加统一 teardown，测试退出时销毁 process groups。

### 7.2 数据并行

- 第一版仅启用 DP，其他并行 size 均为 1。
- 使用 DP rank 而不是 global rank 配置 sampler。
- 所有 epoch 调用 `sampler.set_epoch(epoch)`。
- 明确定义：

```text
global_batch_size = micro_batch_size × num_microbatches × DP
```

- 将模型包装为 MCore DDP；初始可以关闭 bucket overlap。
- loss 按全局有效 token 数归一化，不直接平均不同 rank 的 batch mean。
- validation 的 NLL、token count、aux/z loss 在 DP group 上规约。
- 日志仅由指定 data/model-parallel rank 输出。

### 7.3 验收门槛

- DP=2 与单卡 gradient accumulation 使用相同 global batch 时，loss、梯度和一次更新一致。
- 两个 DP replica 消费不同数据且无遗漏。
- 不允许所有 rank 同时覆盖同一个 checkpoint 文件。

## 8. 里程碑 M2：Tensor Parallel 和词表分片

### 8.1 并行层替换

Attention：

```text
QKV projection:    ColumnParallelLinear
local attention:   local Q/K/V heads
output projection: RowParallelLinear
```

Dense SwiGLU：

```text
gate+up: ColumnParallelLinear(H, 2×FFN)
down:    RowParallelLinear(FFN, H)
```

将现有两个 gate/up Linear 合并为一次投影，输出后再 split。

Embedding/LM Head：

```text
Embedding: VocabParallelEmbedding
LM Head:   ColumnParallelLinear(gather_output=False)
Loss:      VocabParallelCrossEntropy
```

训练期间不 All-Gather 完整 logits；推理接口按需提供 gather 开关。

### 8.2 Attention 改造点

- `n_heads` 改为 global head 数；模块内部保存 local head 数。
- Q/K/V reshape 使用 local head 数。
- MHA 和 GQA 都先实现可整除的 head shard。
- MQA 暂时仅允许 TP=1，直到实现 replicated KV。
- KV cache 元数据必须记录 local KV head layout；训练阶段先不扩展 distributed inference。

### 8.3 Embedding 绑定

- PP=1 时，VocabParallelEmbedding 和 LM Head 可以直接绑定本地 vocab shard。
- PP>1 时首尾 stage 各自保存 shard replica，通过 embedding group 同步梯度。
- 在首尾同步实现完成前，PP>1 配置暂时禁止 `tie_embedding=True`。

### 8.4 验收门槛

- Column/Row Parallel Linear 与完整 Linear 的 forward/dgrad/wgrad 对齐。
- TP=2 Attention、Dense FFN、Embedding、LM Head、CE 分别对齐单卡。
- TP=2 完整模型在聚合 shard 后与 M0 的 loss、梯度和一次 optimizer step 对齐。
- 每个 TP rank 只保存自己的参数 shard。

## 9. 里程碑 M3：Sequence Parallel

### 9.1 激活布局

Transformer block 边界保持：

```text
[S/TP, B, H]
```

每个 Attention/FFN 子层采用：

```text
sequence-sharded input
  -> local LayerNorm
  -> All-Gather sequence
  -> Column Parallel compute
  -> Row Parallel compute
  -> Reduce-Scatter sequence
  -> local Dropout + residual
```

### 9.2 RNG

- 使用 MCore tensor-parallel RNG tracker。
- 明确 data-parallel seed 和 model-parallel seed。
- 保证相同 DP replica 的模型初始化一致。
- 保证 Dropout mask 在 shard 间满足分布式语义，且 resume 后可复现。

### 9.3 验收门槛

- TP2+SP 与 TP2 非 SP 在关闭 Dropout 时数值一致。
- 开启 Dropout 时固定 seed 可重复训练。
- block 边界激活内存相对 TP-only 明显下降。
- TP+EP 配置校验强制开启 SP。

## 10. 里程碑 M4：Pipeline Parallel 和 VPP

### 10.1 Stage-local 模型

将分布式模型构造接口改为：

```python
DistributedTransformer(
    model_config,
    parallel_config,
    pg_collection,
    local_layer_ids,
    pre_process,
    post_process,
    vp_stage,
)
```

- `pre_process=True` 的 stage 构建 Embedding。
- 每个 stage 只构建 `local_layer_ids` 对应的层。
- `post_process=True` 的 stage 构建 FinalNorm、LM Head 和 LM loss。
- 非首 stage 通过 `set_input_tensor()` 接收上游 activation。
- `TransformerOutput.logits` 改为可选，非末 stage 返回 hidden state 或由 schedule 直接管理 activation。

### 10.2 调度

- 将训练循环拆成 MCore 要求的 `forward_step_func`。
- PP=1 使用无流水线 schedule。
- PP>1 第一版使用 non-interleaved 1F1B。
- 通过后再加入 VPP/interleaved 1F1B。
- 定义 warmup、steady、cooldown 的 microbatch 数量并记录 pipeline bubble。

### 10.3 数据和 loss

- token ids 只在 embedding 所在 stage 必需。
- labels 和 loss mask 只在最后 stage 必需。
- attention mask、position ids 选择以下一种一致方案：
  - 在需要的 stage 独立构造；或
  - 从指定 data rank 广播到同一 model replica。
- 不允许每个 PP stage 使用不同 sampler 结果。

### 10.4 MoE auxiliary loss

现有“完整模型末尾汇总所有层 aux loss”的方式不能跨 PP stage 工作。使用以下方案：

- 优先采用 MCore MoE auxiliary-loss autograd scaler，将每个 stage 的 local aux loss 注入本 stage backward graph。
- Router stats 作为无梯度 metrics 单独规约，不能随 activation 放入主 pipeline tensor。

### 10.5 验收门槛

- PP2 的 forward、backward 和一次更新与无 PP microbatch 参考对齐。
- 每个 PP rank 只持有预期层。
- 非交错 1F1B 无 deadlock，并能处理最后一个 microbatch。
- VPP 启用后结果与普通 PP 对齐，pipeline bubble 有可测量下降。
- tied embedding 开启时首尾 stage shard 更新保持一致。

## 11. 里程碑 M5：Expert Parallel 和 Expert Tensor Parallel

### 11.1 保留参考实现

- 现有 `TopKSparseMoE` 和 `GEMM_TopKSparseMoE` 保留为 EP=1 的数值 oracle。
- 分布式训练路径新增 MCore MoE adapter，不直接在现有 `_dispatch_v1/_dispatch_v2` 中混入 collective。

### 11.2 EP 数据流

```text
local tokens
  -> global Top-K router
  -> 按 destination EP rank 排列
  -> EP All-to-All dispatch
  -> 按 local expert 排列
  -> local experts / Grouped GEMM
  -> inverse All-to-All
  -> 恢复 source token 顺序
  -> Top-K 权重加权合并
```

每个 rank 只保存：

```text
num_local_experts = num_experts / EP
gate_up_weight: [num_local_experts, H, 2×FFN_local]
down_weight:    [num_local_experts, FFN_local, H]
```

ETP>1 时再沿 expert hidden 维切分矩阵。

### 11.3 参数和梯度 group

- Attention、Norm、Router 等 dense/replicated 参数使用对应 dense gradient group。
- Expert 权重仅在拥有同一 expert shard 的 EDP group 中同步。
- Expert 内部 tensor shard 使用 ETP group。
- 禁止使用一个 world-group DDP 同步全部参数。

### 11.4 Router 指标和边界输入

- `tokens_per_expert`、importance、aux loss 在配置定义的统计域上规约。
- 支持某个 local expert 收到零 token。
- 支持某个 EP rank 本次收到零 token。
- Top-2 的两个目标 expert 可以位于不同 EP rank。
- 不使用 `.item()` 驱动通信前的 Python 控制流，避免 device synchronization 和 rank 分歧。

### 11.5 验收门槛

- EP2 输出、input gradient、Router/Expert gradient 与单卡全 expert 参考对齐。
- 每个 rank 只保存 `E/EP` 个 experts。
- 零 token、极不均匀负载、Top-2 跨 rank 均不 deadlock。
- TP2+SP+EP2 smoke test 通过。
- ETP2+EP2 的 expert 参数 shard 和梯度规约正确。

## 12. 里程碑 M6：Context Parallel

### 12.1 第一版范围

- 使用 Transformer Engine/MCore Attention，不在当前 eager Attention 中手写完整 ring attention backward。
- 第一版 `cp_comm_type=p2p`；为调试保留 `all_gather` 对照路径。
- 每个 CP rank 持有本地 query/context shard。
- 使用全局 position ids，而不是根据本地 `q_len/k_len` 重新生成位置。

### 12.2 Batch 切分

- 在 DP replica 选择完整样本后，再使用 MCore CP batch helper 沿 sequence 切分。
- token ids、labels、loss mask、position ids、attention mask 必须使用同一 CP 映射。
- causal attention 优先使用 MCore 的负载均衡切分，而不是简单连续切块。
- CP 权重保持复制，因此 dense 参数梯度在 `DP×CP` group 中规约。

### 12.3 Padding 和 Packed Sequence

当前训练数据使用动态右侧 Padding，CP 后可能出现某个 rank 没有有效 token。分两步处理：

1. CP 正确性阶段使用固定长度、无 Padding 的 synthetic batch。
2. 再支持带 loss mask 的固定长度训练样本，并允许某个 rank 的 MoE 有零有效 token。

Packed Sequence、segment boundary loss mask、hybrid CP 作为后续增强，不与基础 CP 同时引入。

### 12.4 验收门槛

- CP2 All-Gather 对照路径与无 CP Attention 对齐。
- CP2 P2P 路径与 All-Gather 路径对齐。
- causal mask 使用全局位置，跨 shard 边界不泄漏未来 token。
- DP2×CP2 的 token-normalized loss 和梯度正确。
- TP2+SP+CP2+PP2 smoke test 通过。

## 13. 里程碑 M7：Distributed Optimizer、梯度处理和 Checkpoint

### 13.1 Distributed Optimizer

- 首先在同步 DP/TP/PP/EP/CP 路径正确后启用。
- 梯度使用 Reduce-Scatter，optimizer 只更新本地 state shard，随后 All-Gather 计算权重。
- 再逐项启用：
  - gradient reduce overlap。
  - parameter gather overlap。
  - TP communication overlap。
  - PP/EP communication overlap。

### 13.2 全局梯度范数

当前 `torch.nn.utils.clip_grad_norm_` 只能看到本地 shard。替换为：

- 排除 TP duplicate 参数。
- dense 与 expert 参数分别在正确 group 求局部平方和。
- 在模型并行域规约得到全局 norm。
- 使用同一个 clipping coefficient 缩放所有 local shards。

### 13.3 Distributed Checkpoint

checkpoint format 升级为 v2，至少记录：

- 模型结构配置。
- 并行拓扑和 rank order。
- 每个 tensor 的 global shape、shard axis、offset 和 replica id。
- optimizer shards。
- scheduler/global step/epoch。
- 每 rank RNG tracker state。
- 数据 sampler state。
- 使用的 PyTorch、CUDA、NCCL、MCore、TE 版本。

增加从当前 format-v1 完整 checkpoint 到 v2 distributed checkpoint 的一次性转换工具。

### 13.4 验收门槛

- 相同拓扑保存/恢复后下一步训练完全可重复。
- 在支持的范围内，TP/PP/EP 拓扑改变后能够 reshard 加载。
- 任意 rank 不会同时覆盖同一文件。
- checkpoint 中 tied embedding 不会重复或丢失 shard。

## 14. 里程碑 M8：组合验证和性能优化

### 14.1 最小测试矩阵

| 场景 | World | TP | SP | PP | CP | EP | ETP | DP/EDP | 目标 |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---|
| 单卡 Dense | 1 | 1 | 否 | 1 | 1 | 1 | 1 | 1 | Golden baseline |
| DP | 2 | 1 | 否 | 1 | 1 | 1 | 1 | 2 | 数据和梯度同步 |
| TP | 2 | 2 | 否 | 1 | 1 | 1 | 1 | 1 | 参数 shard 对齐 |
| TP+SP | 2 | 2 | 是 | 1 | 1 | 1 | 1 | 1 | 激活 shard 对齐 |
| PP | 2 | 1 | 否 | 2 | 1 | 1 | 1 | 1 | 1F1B 调度 |
| EP | 2 | 1 | 否 | 1 | 1 | 2 | 1 | 1 | Token All-to-All |
| TP+SP+EP | 4 | 2 | 是 | 1 | 1 | 2 | 1 | 1 | MoE 组合 |
| CP | 2 | 1 | 否 | 1 | 2 | 1 | 1 | 1 | 长上下文 shard |
| TP+SP+PP+CP | 8 | 2 | 是 | 2 | 2 | 1 | 1 | 1 | Dense 多维组合 |
| 综合 MoE | 按拓扑 | 2 | 是 | 2 | 可选 | 2+ | 1+ | 派生 | 最终 smoke test |

### 14.2 数值验收

每个新维度 size=2 时，必须与上一阶段比较：

- forward hidden states/logits。
- token-normalized LM loss。
- router aux/z loss。
- input gradient。
- 聚合后的参数梯度。
- optimizer step 后聚合权重。

推荐容差：

- FP32、同步算法：优先 `rtol=1e-5, atol=1e-6`，根据 collective 求和顺序记录合理放宽。
- BF16/FP16：单独制定容差，不用低精度结果掩盖实现错误。

### 14.3 性能验收

正确性完成后记录：

- tokens/s 和每 GPU tokens/s。
- peak allocated/reserved memory。
- pipeline bubble 比例。
- TP/DP/PP/EP/CP 通信耗时。
- 通信与计算重叠比例。
- 每个 expert 的 token 数、最大/平均负载比。
- MFU 或可替代的计算利用率指标。

只有在 profiler 能证明收益时才保留复杂 overlap 配置。

## 15. 文件级改造清单

### `configs/zconfig.py`

- 完成 YAML 解析函数。
- 扩展 `ParallelConfig`。
- 增加跨模型/并行配置校验。
- 保持 `ModelConfig` 与运行拓扑解耦。

### `configs/zparser.py` 和 `configs/train.yaml`

- 增加 TP、PP、VPP、CP、EP、ETP、SP 参数。
- 增加 distributed optimizer 和 overlap 参数。
- CLI/YAML 使用同一 dataclass 和校验逻辑。

### `model/ztransformer.py`

- 保留单卡参考路径。
- 提取通用 block/output 协议。
- 不在此文件中直接加入 process-group 条件分支。

### `model/distributed_transformer.py`

- 新增 stage-local 模型。
- 支持 `pre_process/post_process/local_layer_ids/vp_stage`。
- 只构造本 rank 所需模块。

### `model/zattention.py`

- 保留单卡 Attention oracle。
- mask 构造逐步改成显式 query/key position ids，便于 CP 测试。

### `model/mcore_attention.py`

- TP QKV/output projection 适配。
- local head/GQA 配置转换。
- SP/CP/TE Attention 接入。

### `model/zmoe.py`

- 保留 EP=1 参考实现。
- 删除“必须至少一个有效 token”的分布式假设，或仅在参考路径保留。
- 避免 `.item()` 控制需要跨 rank 一致的逻辑。

### `model/mcore_ffn.py`

- Dense TP SwiGLU。
- MCore MoE adapter。
- 将 MCore MoE 输出适配为项目统一的 FFN 输出协议。

### `train.py`

- 最终缩减为配置解析和 `Trainer` 启动入口。
- 移出直接 forward/backward/optimizer step 逻辑。

### `model/mstore.py`

- 保留 format-v1 单卡读取能力。
- 分布式训练改用新的 checkpoint adapter。
- 提供 v1 -> v2 转换入口。

## 16. 风险与控制措施

### 16.1 Collective deadlock

- 所有配置在创建 group 前统一校验。
- 所有 rank 使用完全一致的 group 创建顺序。
- 测试设置有限 timeout。
- collective 前不使用可能在不同 rank 产生不同结果的 Python 分支。

### 16.2 参数被错误同步

- 每个参数标记 shard/replica 语义。
- dense 参数、expert 参数、embedding replica 使用不同 group。
- 测试显式验证“应该相同的 shard 相同、不同 shard 不被平均”。

### 16.3 Loss 缩放错误

- 使用 loss sum 和全局有效 token count。
- PP 只有末 stage 计算 LM loss。
- CP/DP 下避免简单平均 rank-local mean。
- aux loss 的 autograd 注入和 logging reduction 分开处理。

### 16.4 GQA/MQA 切分错误

- 第一版只支持能被 TP 整除的 GQA KV heads。
- MQA TP>1 在 replicated KV 完成前快速失败。
- checkpoint 保存 global head metadata，而不只保存 local shape。

### 16.5 Padding 与 CP/EP 的零 token

- 基础 CP 先使用无 Padding batch。
- EP dispatcher 和 Grouped GEMM 必须接受零 token rank/expert。
- 加入全 PAD shard、单 expert 空载等专项测试。

### 16.6 Checkpoint 不可迁移

- 从第一版 TP 开始记录 global shape 和 shard metadata。
- 不以 global rank 作为唯一 shard 语义。
- 在每个里程碑加入 save/resume 测试，而不是最后补做。

## 17. 实施顺序和依赖关系

```text
M0 单卡 Golden
 └─ M1 Runtime + DP
     └─ M2 TP + Vocab Parallel Loss
         └─ M3 SP
             ├─ M4 PP/VPP
             ├─ M5 EP/ETP
             └─ M6 CP
                  └─ M7 Distributed Optimizer + Checkpoint
                      └─ M8 多维组合 + 性能优化
```

M4、M5、M6 在 M3 完成后可以分别开发，但最终合并前必须通过统一的模型接口、loss 语义和 checkpoint metadata 约束。

## 18. 统一完成定义

某个里程碑只有同时满足以下条件才算完成：

- 实现代码不破坏 `parallel_size=1` 的单卡路径。
- 自动化单元测试和分布式测试通过。
- 与上一个可信参考完成 forward、loss、gradient、optimizer step 对齐。
- 非法配置在 collective 启动前失败。
- checkpoint 能保存并恢复该阶段新增的 shard。
- 日志记录软件版本、硬件、模型配置和并行拓扑。
- 新增通信在 profiler 中可定位，不存在无法解释的同步点。
- 文档记录该阶段已支持和明确不支持的组合。

## 19. 第一批建议任务

- [ ] 完成 `load_model_config()` 和 `model_config_from_dict()`。
- [ ] 扩展并验证 `ParallelConfig`，但默认所有 size 均为 1。
- [ ] 为 Dense/MoE 建立 M0 golden tests。
- [ ] 新增 `parallel/runtime.py`，完成单进程和 Gloo 双进程初始化测试。
- [ ] 新增 `DistributedTransformer` stage-local 构造接口，先在 PP=1 下与当前模型对齐。
- [ ] 接入 VocabParallelEmbedding、TP Linear 和 vocab-parallel CE。
- [ ] 完成 TP2 数值对齐后再开始 SP，不并行推进 CP/EP 通信实现。

## 20. 参考资料

- [Megatron-Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-Core Tensor Parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [Megatron-Core Pipeline Schedules](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html)
- [Megatron-Core Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron-Core Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-Core Distributed Optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)
- [Megatron-LM parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)
