# Transformer 多卡并行开发计划

## 1. 目标

在当前 `transformer.py` 的 Decoder-only Transformer 基础上，按照“单卡训练基线 → MoE → TP → TP+DP → TP+DP+EP”的顺序，逐步完成模型和训练系统的并行化。

每个阶段都必须满足以下要求后才能进入下一阶段：

1. 有可重复执行的自动化测试。
2. 能完成前向、反向、参数更新和保存/加载的完整训练流程。
3. 能与上一阶段进行数值或训练行为对齐。
4. 异常配置能够尽早报错，而不是在分布式通信中挂起。

最终希望支持的并行维度为：

- **TP（Tensor Parallel）**：切分单层中的张量计算。
- **DP（Data Parallel）**：不同模型副本处理不同数据，并同步梯度。
- **MoE（Mixture of Experts）**：用 Router 和多个 Expert 替代普通 FFN。
- **EP（Expert Parallel）**：将不同 Expert 分布到不同设备。

## 2. 总体原则

### 2.1 正确性优先

每增加一种并行方式，都保留一个小规模、确定性的单卡实现作为参考。测试应比较最终输出、loss、输入梯度和关键参数梯度，而不只检查 shape 或程序是否能够启动。

### 2.2 分阶段控制变量

- 增加 MoE 时先保持单卡，不同时引入分布式通信。
- 增加 TP 时先保持一个数据副本和 Dense FFN。
- 增加 DP 时复用已经验证的 TP 模型。
- 增加 EP 时先验证单卡 MoE，再验证 Expert 的跨卡路由。

### 2.3 明确采用哪一种实现路线

开始 TP 之前需要确定以下路线之一：

- **教学实现路线**：基于 `torch.distributed` 手动实现并行 Linear、collective 和进程组，便于理解底层机制。
- **Megatron Core 路线**：将现有 Linear、Embedding、Transformer Layer 和训练调度迁移到 Megatron Core，使用其 TP、DP、MoE、EP 和分布式 checkpoint 能力。

推荐最终训练使用 Megatron Core；如果手写 TP/EP 的主要目的是学习，可以保留为独立实验实现。不要让自定义通信层和 Megatron 通信层在没有清晰边界的情况下混用。

## 3. 阶段一：跑通当前 Transformer 的训练流程

### 3.1 实现内容

- 增加最小训练入口，包括：
  - 构造或加载 token 数据。
  - 生成 `token_ids`、`attention_mask`、`position_ids` 和 labels。
  - 使用 next-token prediction：输入为 `tokens[:, :-1]`，目标为 `tokens[:, 1:]`。
  - 对 PAD 位置设置 `ignore_index`，避免计算无效 loss。
  - 完成 `zero_grad → forward → loss → backward → optimizer.step`。
- 增加随机种子、device、dtype、学习率和模型规模等配置。
- 增加 checkpoint 保存和恢复。
- 对非 PAD token 统计 loss 和 token 数量。
- 保留当前只允许右侧 Padding 的约束，并为错误输入提供明确错误信息。

### 3.2 测试

- RoPE 的 shape、位置变化和 dtype/device 测试。
- causal mask 测试：位置 `i` 不能读取未来位置。
- padding mask 测试：有效 token 不能读取 PAD key。
- 不允许左侧 Padding 和全 PAD 样本的测试。
- Transformer 前向输出 shape：`[B, L, vocab_size]`。
- 单步训练后至少一个可训练参数发生变化。
- 在固定小数据集上训练若干步，loss 有明显下降。
- checkpoint 恢复后，固定输入的 logits 与保存前一致。
- CPU 必测；CUDA 可用时增加 GPU 测试。

### 3.3 完成标准

- 单条命令可以执行 smoke training。
- 自动化测试能够稳定通过。
- 小数据集可以被模型过拟合，证明训练链路正确。
- 无 PAD、带右侧 PAD 两种输入都能正常训练。

## 4. 阶段二：将 FFN 改造为 MoE

### 4.1 模型结构

将部分或全部 TransformerBlock 中的 SwiGLU FFN 替换为 MoE：

```text
x → Router → Top-K Experts → 按路由权重合并 → output
```

第一版建议采用以下最小配置：

- Expert：复用当前 `SwiGLU`。
- Router：`Linear(d_model, num_experts, bias=False)`。
- 路由策略：先实现 Top-1，再扩展到 Top-2。
- Expert 数量：测试时使用 2 或 4。
- 第一版采用 dropless 路由，不丢弃 token。
- 增加 load-balancing auxiliary loss，避免少数 Expert 长期过载。
- Padding token 不应进入 Expert 负载统计，也不应贡献辅助 loss。

MoE 层建议返回：

```python
output, router_aux_loss, router_stats
```

总 loss 为：

```text
total_loss = language_model_loss + aux_loss_weight * router_aux_loss
```

### 4.2 测试

- Router logits、Top-K index、Top-K weight 的 shape 测试。
- 每个有效 token 恰好被分发到 K 个 Expert。
- Top-K 权重归一化测试。
- Padding token 不计入路由统计。
- 手工构造 Router 权重，验证 token 被发送到预期 Expert。
- MoE 输出与手工逐 token Expert 计算结果对齐。
- Router 和被选中的 Expert 能够收到梯度。
- 单 Expert、Top-1 时与普通 SwiGLU FFN 数值对齐。
- MoE 模型完成训练 smoke test，并在小数据集上降低 loss。

### 4.3 完成标准

- `num_experts=1, top_k=1` 可以作为 Dense FFN 的等价退化配置。
- 多 Expert 下能够完成稳定训练。
- 日志中可以观察每个 Expert 的 token 数量、负载比例和 auxiliary loss。
- checkpoint 能够保存和恢复 Router 与所有 Expert 参数。

## 5. 阶段三：增加 Tensor Parallel（TP）

### 5.1 进程和切分方式

创建 TP process group。第一版只支持单个 TP group，即：

```text
world_size = tp_size
dp_size = 1
ep_size = 1
```

需要实现或接入以下 TP 模块：

- Vocab Parallel Embedding。
- Column Parallel Linear。
- Row Parallel Linear。
- Attention QKV 投影的 Head/输出维度切分。
- Attention output projection 的输入维度切分。
- SwiGLU gate/up 的输出维度切分。
- SwiGLU down projection 的输入维度切分。
- 并行 LM Head 和 vocab-parallel cross entropy。

推荐同时实现 Sequence Parallel，或者明确记录第一版 TP 中哪些激活保持复制状态。

### 5.2 配置约束

初始化分布式环境时提前检查：

- `world_size % tp_size == 0`。
- `d_model % tp_size == 0`。
- `num_heads % tp_size == 0`。
- `hidden_dim % tp_size == 0`。
- vocab size 能够均匀切分，或明确采用 padding 后的 padded vocab size。
- 所有 rank 使用一致的模型配置和随机种子策略。

### 5.3 测试

使用 `torchrun` 启动多进程测试，至少覆盖 TP=1 和 TP=2：

- Column/Row Parallel Linear 与完整 `nn.Linear` 数值对齐。
- TP Attention 与单卡 Attention 的输出、输入梯度和参数梯度对齐。
- TP SwiGLU 与单卡 SwiGLU 对齐。
- 并行 Embedding、LM Head 和 cross entropy 与单卡结果对齐。
- 完整 Transformer 在固定权重和输入下与单卡 logits/loss 对齐。
- 多步训练后，聚合得到的参数与单卡参考训练结果在容差内一致。
- 不满足整除条件时所有 rank 都快速失败，不发生通信挂起。

### 5.4 完成标准

- TP=1 与原模型行为一致。
- TP=2 能够完成完整训练、反向传播和 checkpoint 恢复。
- TP rank 只保存属于自己的参数分片。
- 分布式 checkpoint 明确记录 TP 拓扑和分片信息。

## 6. 阶段四：在 TP 基础上增加 DP/DDP

### 6.1 进程拓扑

定义二维进程网格：

```text
world_size = tp_size × dp_size
```

例如 `world_size=8, tp_size=2, dp_size=4`：

```text
TP group 0: ranks [0, 1]
TP group 1: ranks [2, 3]
TP group 2: ranks [4, 5]
TP group 3: ranks [6, 7]

DP group for TP-rank 0: ranks [0, 2, 4, 6]
DP group for TP-rank 1: ranks [1, 3, 5, 7]
```

DDP 必须使用 DP process group，而不是默认 world group；否则会错误地同步不同 TP 分片。

### 6.2 实现内容

- DistributedSampler 或等价数据切分机制。
- 每个 DP replica 获取不同 microbatch。
- 同一 TP group 内的 rank 获取相同输入。
- 只在对应 DP group 中同步相同 TP 分片的梯度。
- 正确处理 gradient accumulation 和 `no_sync`。
- 定义：

```text
global_batch_size = micro_batch_size × grad_accum_steps × dp_size
```

- 只由指定 rank 写日志和 checkpoint，避免文件竞争。
- 汇总 loss 时按有效 token 数加权，而不是直接平均各 rank 的 batch loss。

### 6.3 测试

- 验证 TP group、DP group 的成员关系符合预期。
- 验证不同 DP replica 获得不同样本。
- 验证同一 TP group 内输入一致。
- 一步优化后，同一 DP group 中对应参数分片保持一致。
- TP=1、DP=2 与普通 DDP 结果对齐。
- TP=2、DP=2 与相同 global batch 的单卡梯度累积结果对齐。
- checkpoint 只写一次，恢复后所有 rank 参数正确。
- 任意 rank 发生异常时，测试能够超时退出而不是无限等待。

### 6.4 完成标准

- TP=2、DP=2 能稳定完成多步训练。
- 数据无重复、无遗漏，最后一个不完整 batch 的行为有明确约定。
- global loss、梯度和单卡参考结果在预期容差内一致。
- 日志明确显示 world size、TP/DP size、rank 和 process group。

## 7. 阶段五：MoE 支持 Expert Parallel（EP）

### 7.1 进程拓扑

在进入实现前固定并记录 TP、DP、EP 的 rank 映射。若使用 Megatron Core，应直接采用其 model-parallel state 和 process group；若手写实现，应先为进程网格编写独立测试。

EP 的基本目标是：

- 每个 EP rank 只保存一部分 Expert。
- Router 根据全局 Expert ID 选择 Expert。
- Token 通过 All-to-All 被发送到目标 Expert 所在 rank。
- Expert 计算完成后，通过反向 All-to-All 返回原 rank。
- 原 rank 根据 Top-K 权重合并结果并恢复 token 原顺序。

Dense 参数与 Expert 参数的梯度同步组不同：

- Attention、Norm、Router 等共享参数按照模型拓扑在正确的 DP/复制组中同步。
- Expert 参数只在保存同一个 Expert 分片的 expert-data-parallel group 中同步。
- 不同 Expert 的参数绝不能被普通 world-group DDP 错误平均。

### 7.2 实现内容

- 全局 Expert ID 到 `(ep_rank, local_expert_id)` 的映射。
- Token permutation、dispatch、local grouped expert computation、combine 和 unpermutation。
- Top-1/Top-2 All-to-All token dispatcher。
- 每个 rank 的 send/receive split size 交换。
- 变长 token 数量和某些 Expert 接收零 token 的处理。
- Router auxiliary loss 的跨相关进程组统计。
- Expert 参数的分片 checkpoint 和重分发约定。
- 性能优化放在正确性之后：Grouped GEMM、通信与计算重叠、容量限制等。

若使用 Megatron Core，优先使用其 MoE Router、token dispatcher、Grouped GEMM、EP process group 和 distributed checkpoint，不重复实现底层 collective。

### 7.3 测试

- `num_experts=1, EP=1` 与 Dense FFN 退化配置对齐。
- 多 Expert、EP=1 与阶段二的单卡 MoE 对齐。
- 固定 Router 路由，验证每个 token 到达正确 EP rank 和本地 Expert。
- EP=2 的输出、loss、输入梯度与单卡全 Expert MoE 对齐。
- Expert 收到零 token、负载极不均衡、Top-2 跨两个 rank 等边界测试。
- TP=1、DP=1、EP=2 的基础组合测试。
- TP=2、DP=2、EP=2 的最终组合 smoke test。
- 验证共享参数和 Expert 参数分别在正确进程组中同步。
- 保存并恢复分布式 MoE checkpoint 后结果一致。

### 7.4 完成标准

- 每个 EP rank 只持有预期的本地 Expert。
- 跨 rank 路由前后 token 数量、顺序和权重均正确。
- EP 输出与非 EP 的参考 MoE 在容差内一致。
- TP+DP+EP 能稳定训练，不发生 collective shape mismatch 或 deadlock。
- 日志可以观察全局及各 rank 的 Expert 负载分布。

## 8. 建议的测试矩阵

| 阶段 | world size | TP | DP | EP | 主要验证目标 |
|---|---:|---:|---:|---:|---|
| 单卡 Dense | 1 | 1 | 1 | 1 | 训练基线、mask、checkpoint |
| 单卡 MoE | 1 | 1 | 1 | 1 | Router、Expert、辅助 loss |
| TP | 2 | 2 | 1 | 1 | 分片 Linear、Attention、FFN |
| DDP | 2 | 1 | 2 | 1 | 数据切分和梯度同步 |
| TP + DP | 4 | 2 | 2 | 1 | 二维进程组和端到端训练 |
| EP | 2 | 1 | 1 | 2 | All-to-All Expert 路由 |
| TP + DP + EP | 按选定拓扑 | 2 | 2 | 2 | 最终组合和 checkpoint |

CI 中没有足够 GPU 时，可以保留 CPU/Gloo 的小张量通信测试；正式验收仍需使用 CUDA/NCCL 完成至少一次多卡测试。

## 9. 建议的代码组织

随着功能增加，不建议继续把所有内容放在单个 `transformer.py` 中。可以逐步拆分为：

```text
transformer.py              # Dense 模型或兼容入口
model/
  attention.py
  rope.py
  ffn.py
  moe.py
  transformer.py
parallel/
  parallel_state.py         # TP/DP/EP process groups
  tensor_parallel.py
  expert_parallel.py
training/
  train.py
  checkpoint.py
tests/
  test_transformer.py
  test_moe.py
  test_tensor_parallel.py
  test_data_parallel.py
  test_expert_parallel.py
  distributed/
    run_tp_test.py
    run_tp_dp_test.py
    run_ep_test.py
```

如果采用 Megatron Core，上述 `parallel/` 中应主要放置配置、适配和初始化代码，而不是重新实现 Megatron 已提供的 collective。

## 10. 推荐里程碑

- [ ] **M1：Dense 单卡基线**——测试、训练、过拟合小数据、checkpoint 全部通过。
- [ ] **M2：单卡 MoE**——Top-K 路由、辅助 loss、训练和 checkpoint 全部通过。
- [ ] **M3：TP**——TP=2 与单卡参考模型完成输出和梯度对齐。
- [ ] **M4：TP+DP**——TP=2、DP=2 完成多步训练和恢复。
- [ ] **M5：TP+DP+EP**——MoE 跨卡路由、梯度同步和分布式 checkpoint 全部通过。
- [ ] **M6：性能验收**——在正确性测试通过后测量吞吐、显存、通信时间和 Expert 负载。
- [ ] 学习并同步 MLA, DSpark 等 DeepSeek 的推理优化技术
- [ ] 适时添加推理功能, 更直观地检测模型的训练效果

## 11. 每个里程碑的统一验收记录

每次完成里程碑时记录：

- Git commit 或版本标识。
- PyTorch、CUDA、NCCL、Megatron Core 版本。
- GPU 型号和数量。
- 模型配置及 TP/DP/EP 配置。
- 固定随机种子。
- 测试命令与测试结果。
- 单步及多步 loss。
- 峰值显存、tokens/s 和 step time。
- 已知限制和下一阶段需要解决的问题。

这样可以区分数值正确性问题、分布式拓扑问题和性能问题，也方便后续升级 PyTorch 或 Megatron Core 时进行回归验证。
