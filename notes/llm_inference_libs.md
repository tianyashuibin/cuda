# LLM 推理优化核心 CUDA 库整理

> 聚焦 LLM 推理优化方向，只收录实际会用到的库和工具。

---

## 1. GEMM — 推理计算量的绝对核心 (~70%)

| 库 | 定位 | 说明 |
|---|------|------|
| **cuBLAS** | NVIDIA 官方 BLAS | 开箱即用的高性能 GEMM，闭源，大部分框架的默认后端 |
| **cuBLASLt** | cuBLAS 轻量版 | 比 cuBLAS 更灵活：支持混合精度（FP16/INT8/FP8）、自定义 layout、epilogue fusion（GEMM+Bias+Activation 一步完成） |
| **CUTLASS** | NVIDIA 开源 GEMM 模板库 | 可深度定制的 Tensor Core kernel，理解 tiling/warp/pipeline 的最佳学习材料；TensorRT-LLM 底层大量使用 |

**实际选择：**
- 快速出活 → cuBLAS / cuBLASLt
- 需要定制（量化 kernel、特殊数据类型、epilogue 融合）→ CUTLASS
- 学习 Tensor Core 编程 → 从 CUTLASS 的 examples 入手

---

## 2. Attention — 推理延迟的关键瓶颈

| 库 | 定位 | 说明 |
|---|------|------|
| **FlashAttention** | Tri Dao 开源 | Fused tiling attention，减少 HBM 读写；几乎是行业标准；支持 FP16/BF16 |
| **FlashInfer** | 专注 LLM Serving | 针对推理场景深度优化：PageAttention、各种 decode kernel（GQA/MQA）、prefix caching；vLLM/SGLang 的 attention 后端 |
| **FlashMLA** | DeepSeek 开源 | 针对 Multi-head Latent Attention (MLA) 结构优化的 kernel，DeepSeek-V2/V3 使用 |

**实际选择：**
- Prefill 阶段 → FlashAttention
- Decode / Serving 场景 → FlashInfer（对 KV Cache 管理、动态 batch 支持更好）
- DeepSeek 系列模型 → FlashMLA

---

## 3. 量化推理 — 降低显存和延迟

| 库/方案 | 量化类型 | 说明 |
|---------|---------|------|
| **Marlin** | W4A16 | IST-DASLab 开源，目前最快的 W4A16 GEMM kernel；vLLM 默认使用 |
| **AWQ** | W4A16 | 激活感知的权重量化，有配套 CUDA kernel |
| **GPTQ** | W4A16 | 基于二阶信息的权重量化，AutoGPTQ 提供 kernel |
| **bitsandbytes** | INT8/FP4/NF4 | Hugging Face 集成，QLoRA 用这个；推理性能一般，更偏训练侧 |
| **FP8** | W8A8 | Hopper (H100) 原生支持，cuBLASLt/CUTLASS 直接调用，精度损失小 |

**实际选择：**
- H100/H200 → 优先 FP8（原生支持，无需额外 kernel）
- A100/消费卡 → W4A16（Marlin/AWQ）
- 需要最极致压缩 → W4A4 方案（目前还在快速发展中）

---

## 4. 多卡通信 — 分布式推理必备

| 库 | 说明 |
|---|------|
| **NCCL** | NVIDIA 官方集合通信库，AllReduce/AllGather/ReduceScatter；张量并行、流水线并行的底层通信 |
| **MSCCL / MSCCL++** | 微软优化版，支持自定义通信拓扑 |
| **NIXL** | NVIDIA 新推出的跨节点推理通信库，针对 disaggregated serving（PD 分离） |

**实际选择：**
- 单机多卡 → NCCL 足够
- 多机大规模部署 → NCCL + 网络拓扑优化
- PD 分离架构 → 关注 NIXL

---

## 5. 基础并行原语

| 库 | 说明 |
|---|------|
| **CUB** | NVIDIA 开源，高性能 block/device 级原语（reduce、scan、sort、radix select）；Top-k sampling、Softmax reduce 等底层会用到 |
| **Thrust** | CUB 的高层封装，STL 风格接口；快速原型开发用 |

---

## 6. 性能分析工具 — 优化的前提

| 工具 | 说明 |
|------|------|
| **Nsight Systems** (`nsys`) | 系统级时间线：找到哪个 kernel 慢、CPU/GPU 是否有 gap |
| **Nsight Compute** (`ncu`) | Kernel 级分析：roofline、occupancy、memory throughput、指令瓶颈 |
| **NVTX** | 代码打标注，配合 Nsight 定位具体代码段 |
| **CUDA Graphs** | 不是分析工具，但减少 kernel launch 开销（推理框架常用） |

---

## 7. Kernel 开发工具

| 工具 | 说明 |
|------|------|
| **Triton** | OpenAI 的 Python GPU 编程 DSL，写 kernel 比原生 CUDA 快很多；适合快速实现 fused kernel 原型 |
| **CUDA C/C++** | 追求极致性能还是得手写；CUTLASS 基于此 |
| **CuTe** | CUTLASS 3.x 引入的布局抽象库，描述 Tensor Core 数据搬运的核心工具 |

---

## 8. 主流推理引擎 — 上述库的集大成者

| 引擎 | 核心特点 | 底层依赖 |
|------|---------|---------|
| **vLLM** | PagedAttention，高吞吐 serving | FlashAttention/FlashInfer + Marlin + CUTLASS + NCCL |
| **SGLang** | RadixAttention，高效调度 | FlashInfer + CUTLASS + NCCL |
| **TensorRT-LLM** | NVIDIA 官方，极致单卡性能 | CUTLASS + cuBLAS + FP8 + CUDA Graphs + NCCL |
| **llama.cpp** | CPU/GPU 混合，端侧部署 | 自研 CUDA kernel，量化为主 |

---

## 9. 学习路线建议

```
                    基础                          核心                          进阶
              ┌──────────────┐          ┌──────────────────┐          ┌──────────────────┐
              │ CUDA C 基础   │          │ CUTLASS / CuTe   │          │ 手写 fused kernel │
   GEMM 线 ── │ Shared Memory │───────▶  │ Tensor Core 编程  │───────▶  │ 量化 GEMM kernel  │
              │ cuBLAS 调用   │          │ Tiling 策略       │          │ FP8/INT4 定制     │
              └──────────────┘          └──────────────────┘          └──────────────────┘

              ┌──────────────┐          ┌──────────────────┐          ┌──────────────────┐
              │ Attention 原理│          │ FlashAttention   │          │ FlashInfer 源码   │
Attention 线  │ Online Softmax│───────▶ │ 源码阅读           │───────▶  │ 定制 decode kernel│
              │ Triton 实现   │          │ Tiling + 融合     │          │ PageAttention    │
              └──────────────┘          └──────────────────┘          └──────────────────┘

              ┌──────────────┐          ┌──────────────────┐          ┌──────────────────┐
              │ Nsight 基本用法│          │ 分析实际 kernel   │          │ 读引擎源码        │
 工程线 ────   │ nvtx 打标注    │───────▶ │ roofline 分析     │───────▶  │ vLLM / SGLang    │
              │ CUDA Graphs  │          │ 瓶颈定位          │          │ TensorRT-LLM     │
              └──────────────┘          └──────────────────┘          └──────────────────┘
```

---

## 10. 速查：库与算子的对应关系

```
LLM 推理一次 forward 的数据流：

Token ──▶ Embedding Lookup
              │
              ▼
    ┌─── Transformer Layer × N ───┐
    │                              │
    │  RMSNorm ◄── 手写 kernel     │
    │     │                        │
    │  QKV Proj ◄── cuBLAS/CUTLASS │
    │     │                        │
    │  RoPE ◄── 手写 kernel        │
    │     │                        │
    │  Attention ◄── FlashAttention│
    │     │          / FlashInfer  │
    │  O Proj ◄── cuBLAS/CUTLASS   │
    │     │                        │
    │  RMSNorm ◄── 手写 kernel     │
    │     │                        │
    │  FFN (SwiGLU) ◄── CUTLASS   │
    │     │              + 融合    │
    └─────┼────────────────────────┘
          │
          ▼
    LM Head ◄── cuBLAS/CUTLASS
          │
          ▼
    Sampling ◄── CUB (Top-k sort)
```
