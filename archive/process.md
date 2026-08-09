# TODO next

- 参照代码中的 TODO 说明.
- [优先级下降] 合并训练 MHA、Prefill MHA、Decode MHA
- triton 融合算子实现.
- FSDP2 并行实现, 支持 FSDP2 之后引入 Megatron-Core 实现多维并行.

# ON GOING ...

1. 统一 Attention 初始化、输入、输出接口.
2. Config 中增加 Segment Attention Mask 配置, 区分和当前 Attention.
3. TransformerBlock 中根据配置路由到支持 Segment Attention Mask 的 Attention.
4. 新增 segment_attention.py, 使用统一后的 Attention 接口规范.
5. 增加 Attention 工厂, 融合 TransformerBlock 中的路由功能, 支持创建 Segment Attention 和当前 Attention.

# Waiting ...

1. Megatron-Core Tensor-Parallel
2. Megatron-Core Data-Parallel
3. Megatron-Core Expert-Parallel
4. To Learn about Context-Parallel
5. Triton Kernel !! More Efficient to Go

- 打通推理链路.
