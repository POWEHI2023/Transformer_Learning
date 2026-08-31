# vLLM 推理服务框架

## 启动方式

### 一个逻辑 vLLM 服务的分布式 Worker

### 多个独立 vLLM 服务的请求级负载均衡

## 推理并发

### 分布式 Worker 中的并发 - TP/PP/PCP/DCP

### 请求级负载均衡 - DP

## vLLM EngineCore 中的调度

### Continuous Batching Scheduler

#### Dispatched Prefill & Decode

#### Fused Prefill & Decode

### Block-Wise KV Cache ModelRunner

### EngineCoreProc 中启动的 Poll 异步请求处理 - Input/Output/BusyLoop

## vLLM Paged Attention & FlashInfer

### Paged Attention

### FlashInfer

### Flash Attention - Online Softmax

### Basic Attention - MHA/MQA/GQA/MLA/Linear Attention

#### MHA/MQA/GQA

#### Multi-Head Latent Attention

#### Linear Attention/ Gated Delta Attention

## KV Connector - Remote KV Cache