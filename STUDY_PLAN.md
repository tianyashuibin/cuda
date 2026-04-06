# LLM 推理优化转型学习计划

> 目标：3-6 个月内达到阶段二水平，具备 LLM 推理优化岗位的面试竞争力
>
> 时间预算：工作日 2h/天 + 周末 5h ≈ **15h/周**
>
> 背景：后端开发（推荐系统），有 C++ 功底，GPU 架构已有基础理解

---

## 一、当前进度评估

### 已完成 ✅

| 领域 | 完成内容 | 对应阶段 |
|------|---------|---------|
| CUDA GEMM 优化 | shared mem → reg tiling → bank conflict fix → double buffer → warp tiling → Tensor Core WMMA，7 步完整链路 | 阶段一核心 |
| Triton 编程 | vector_add、softmax、matmul、flash_attention、layernorm、quantize_matmul 共 6 个 sample | 阶段一核心 |
| 性能分析工具 | Nsight Systems / Compute 使用、NVTX 标注 | 阶段一 |
| CUDA 基础 | stream、CUDA Graph、向量加法 | 阶段一基础 |
| 理论知识 | LLM 算子整理、算子融合原理、PyTorch 算子对应关系、推理优化核心库 | 阶段一 |

### 未完成 ❌

| 领域 | 缺失内容 | 对应阶段 |
|------|---------|---------|
| CUTLASS / CuTe | 未接触，这是阶段二最核心的内容 | 阶段二核心 |
| FlashAttention CUDA 源码 | Triton 版写过，但 CUDA 原版未读 | 阶段二核心 |
| 量化 GEMM（CUDA 级） | Triton 版写过，CUDA 级的 Marlin/W4A16 kernel 未接触 | 阶段二 |
| 推理引擎源码 | vLLM / SGLang 未深入 | 阶段二 |
| 开源贡献 | 无 kernel 级 PR | 阶段二毕业标志 |
| cuBLAS/cuBLASLt 高级用法 | 混合精度、epilogue fusion 等 | 阶段一补全 |

### 结论

> **阶段一已完成约 80%**，剩余 cuBLASLt 高级用法需补全。
> 阶段二从零开始，是接下来的主要战场。

---

## 二、阶段一收尾（第 1-2 周）

目标：补全阶段一剩余内容，为阶段二打下基础。

### 第 1 周：cuBLASLt 混合精度 + GEMM 优化收尾

**工作日（10h）：**
- 用 cuBLASLt API 实现 FP16 输入 → FP32 累加 的 GEMM（2h）
- 用 cuBLASLt 的 epilogue 实现 GEMM + Bias + ReLU 融合（3h）
- 把你的手写 GEMM（reg tiling 版）与 cuBLAS 做详细 Nsight Compute 对比分析，找出性能差距的根因（3h）
- 整理笔记：cuBLASLt API 使用指南（2h）

**周末（5h）：**
- 用 Nsight Compute 的 roofline 模型分析你的 GEMM 各版本，画出优化曲线图（3h）
- 回顾 OPTIMIZATION_ROADMAP.md 中 step5 warp tiling 的退步问题，尝试定位原因（2h）

**产出：**
- `cuda_samples/cublaslt_demo.cu` — cuBLASLt 混合精度 + epilogue 示例
- `notes/cublaslt_guide.md` — API 使用笔记
- GEMM 各版本的 roofline 对比图

### 第 2 周：CUTLASS 环境搭建 + 入门

**工作日（10h）：**
- 编译 CUTLASS，跑通官方 example：`00_basic_gemm`（3h，编译 CUTLASS 可能踩坑）
- 读懂 `00_basic_gemm` 的代码结构：理解 CUTLASS 的 Gemm template 参数含义（4h）
- 修改 `00_basic_gemm` 的参数：改 tile size、数据类型、layout，观察性能变化（3h）

**周末（5h）：**
- 读 CUTLASS 文档中的 "Fundamental types" 和 "GEMM API" 章节（3h）
- 对比你的手写 GEMM 和 CUTLASS basic GEMM 的 Nsight Compute 报告（2h）

**产出：**
- CUTLASS 编译成功并能跑通 example
- `notes/cutlass_basics.md` — CUTLASS 架构和 API 笔记

---

## 三、阶段二详细计划（第 3-20 周）

### 第一阶段：CUTLASS 深入（第 3-6 周）

#### 第 3 周：CUTLASS 2.x GEMM 深入

- 读 CUTLASS 2.x 的 GEMM 实现：threadblock tile → warp tile → instruction tile 的三级 tiling（8h）
- 对照你手写 GEMM 的 tiling 理解 CUTLASS 是如何做的（4h）
- 整理笔记：CUTLASS tiling hierarchy（3h）

**产出：**
- `notes/cutlass_tiling.md` — 三级 tiling 详解，对比手写 GEMM

#### 第 4 周：CuTe 入门

- 读 CUTLASS 3.x 文档中的 CuTe 部分（5h）
- 跑通 CuTe 的 tutorial examples（5h）
- 理解 Layout、Tensor、copy/mma 抽象（5h）

**产出：**
- `notes/cute_guide.md` — CuTe 核心概念笔记

#### 第 5 周：CUTLASS 3.x GEMM

- 读 CUTLASS 3.x 的一个 GEMM example，理解与 2.x 的区别（7h）
- 重点理解：CollectiveMainloop、TiledMma、GmemTiledCopy（5h）
- 修改参数跑 benchmark（3h）

**产出：**
- 能修改 CUTLASS 3.x GEMM example 的参数并跑通
- 阶段检查：能口头解释 CUTLASS 的 GEMM 从 global memory 到 Tensor Core 的完整数据流

#### 第 6 周：CUTLASS 实战 — 写一个自定义 epilogue

- 用 CUTLASS 实现 GEMM + Bias + SiLU 的 fused kernel（10h）
- 与 cuBLASLt 的 epilogue 版本做性能对比（3h）
- 整理文档（2h）

**产出：**
- `cuda_samples/cutlass_fused_gemm.cu` — 自定义 epilogue 实现
- **里程碑 1 达成：能用 CUTLASS 写定制 GEMM**

---

### 第二阶段：FlashAttention 源码（第 7-10 周）

#### 第 7 周：FlashAttention 算法精读

- 重读 FlashAttention 1 & 2 论文，重点理解 IO 复杂度分析（5h）
- 对照你已经写的 Triton 版 FlashAttention，理解每一步对应论文的哪个公式（5h）
- 手推 online softmax 的数学推导，确保完全理解（5h）

**产出：**
- `notes/flash_attention_deep_dive.md` — 算法推导 + Triton 实现对照

#### 第 8-9 周：FlashAttention CUDA 源码阅读

- clone FlashAttention repo，定位 forward kernel 入口（2h）
- 逐函数读 forward kernel：理解 GEMM（QK^T 和 PV）如何用 CUTLASS/CuTe 实现（12h）
- 理解 tiling 策略：outer loop over KV blocks、inner loop over Q blocks（6h）
- 理解 causal mask 的实现方式（3h）
- 理解 softmax rescaling 的实现（3h）
- 对比 CUDA 版和你的 Triton 版的差异（4h）

**产出：**
- `notes/flash_attention_cuda_source.md` — 源码阅读笔记，标注关键函数和行号
- **里程碑 2 达成：能讲清楚 FlashAttention CUDA 版的完整实现**

#### 第 10 周：FlashAttention 变体

- 读 FlashInfer 的 decode attention kernel（理解与 prefill 的区别）（8h）
- 理解 PagedAttention 的 KV Cache 管理方式（4h）
- 对比 FlashAttention / FlashInfer / FlashMLA 的设计选择（3h）

**产出：**
- `notes/attention_variants.md` — 各 attention kernel 变体对比

---

### 第三阶段：量化 Kernel（第 11-14 周）

#### 第 11 周：量化基础

- 读 AWQ / GPTQ 论文的量化部分（不需要读完整论文，重点是量化算法）（5h）
- 理解 per-channel / per-group quantization 的区别（3h）
- 理解 W4A16：权重 INT4 打包格式、dequantize 的计算过程（4h）
- 整理笔记（3h）

**产出：**
- `notes/quantization_basics.md` — 量化原理和数据格式

#### 第 12-13 周：Marlin Kernel 源码

- clone Marlin repo，跑通 benchmark（3h）
- 读 Marlin 的 W4A16 GEMM kernel 源码（15h，这个很有挑战性）
  - 理解 INT4 权重的 packing/unpacking
  - 理解如何利用 Tensor Core 做混合精度计算
  - 理解 async copy + software pipelining
- 与你的 Triton quantize_matmul 做对比（3h）
- 尝试修改 Marlin kernel 的一个参数（如 group size）并验证正确性（4h）
- 整理笔记（5h）

**产出：**
- `notes/marlin_kernel_analysis.md` — Marlin 源码分析
- **里程碑 3 达成：能读懂并修改一个生产级量化 kernel**

#### 第 14 周：FP8 GEMM（如果有 Hopper 硬件）

- 用 CUTLASS 3.x 实现 FP8 GEMM（如有 H100）（8h）
- 或：用 Triton 实现 FP8 GEMM 并在 Colab（T4/A100）上跑（8h）
- 理解 FP8 的 E4M3 / E5M2 格式和 scaling 策略（4h）
- 整理笔记（3h）

**产出：**
- FP8 GEMM 实现（CUTLASS 或 Triton）
- `notes/fp8_guide.md`

---

### 第四阶段：推理引擎源码 + 开源贡献（第 15-20 周）

#### 第 15-16 周：vLLM 源码阅读

- clone vLLM，跑通一个基本的 serving demo（3h）
- 读 vLLM 的 model runner：理解一次 forward 的完整调用链（8h）
- 读 vLLM 的 attention backend：理解如何调用 FlashInfer / FlashAttention（7h）
- 读 vLLM 的 quantization 模块：理解如何集成 Marlin / AWQ kernel（5h）
- 读 vLLM 的 scheduler：理解 continuous batching（4h）
- 整理笔记（3h）

**产出：**
- `notes/vllm_architecture.md` — vLLM 核心模块分析

#### 第 17-18 周：找到贡献切入点 + 提 PR

- 浏览 vLLM / SGLang 的 GitHub Issues，找 `good first issue` 或 kernel 相关 bug（5h）
- 可能的切入点：
  - kernel 性能 regression 修复
  - 新模型的 attention 适配
  - 量化 kernel 的 bug fix
  - benchmark / profiling 工具改进
- 实现并提交 PR（20h）
- 根据 review 反馈修改（5h）

**产出：**
- **里程碑 4 达成：向 vLLM 或 SGLang 提交一个 PR（哪怕是小的）**

#### 第 19-20 周：综合项目 + 面试准备

- 做一个端到端的推理优化 demo（15h）：
  - 选一个开源 LLM（如 Llama-3-8B）
  - 用 vLLM serving
  - 用 Nsight 分析瓶颈
  - 尝试优化一个 kernel（如 fused RMSNorm + residual）
  - 写出完整的分析报告：瓶颈在哪、为什么、怎么优化、效果如何
- 准备面试材料（10h）
- 整理 GitHub 项目（5h）

**产出：**
- 端到端优化 demo + 分析报告
- **里程碑 5 达成：有一个可展示的完整优化案例**

---

## 四、里程碑检查清单

| # | 里程碑 | 验证标准 | 目标周 |
|---|--------|---------|-------|
| 1 | CUTLASS 定制 GEMM | 能用 CUTLASS 写 GEMM + 自定义 epilogue 并跑出正确结果 | 第 6 周 |
| 2 | FlashAttention 源码 | 能在白板上画出 FA CUDA 版的完整 tiling + 数据流，解释每一步 | 第 9 周 |
| 3 | 量化 Kernel | 能读懂 Marlin 源码并修改参数，理解 INT4 pack/unpack + Tensor Core 调用 | 第 13 周 |
| 4 | 开源贡献 | 向 vLLM / SGLang 提交至少一个被合入的 PR | 第 18 周 |
| 5 | 端到端优化 | 有一个完整的 "发现瓶颈 → 分析 → 优化 → 验证" 案例 | 第 20 周 |

---

## 五、每周时间分配模板

```
周一 (2h)：阅读源码 / 论文（纯输入，不写代码）
周二 (2h)：动手写代码 / 复现
周三 (2h)：动手写代码 / 调试
周四 (2h)：Nsight 分析 / 性能对比
周五 (2h)：整理笔记 / 复盘本周进度
周末 (5h)：集中攻克本周最难的部分（通常是源码阅读或 CUTLASS 实验）
```

**注意事项：**
- 周一读的内容决定了周二到周四干什么，所以**周一要规划好本周目标**
- 周五的笔记整理不是浪费时间，它帮你巩固理解，也是将来面试时的复习材料
- 周末的 5h 建议连续使用，不要切碎 — CUTLASS 和源码阅读需要沉浸式的长时间块
- 如果某周卡住了（比如 CUTLASS 编译问题），**不要死磕超过 4h**，上 GitHub Issues / Stack Overflow 找答案或者暂时跳过

---

## 六、卡住时的应对策略

| 卡住的地方 | 应对策略 |
|-----------|---------|
| CUTLASS 编译不过 | 先用 Colab / Docker（NVIDIA NGC 容器），不要在本地环境上浪费太多时间 |
| CUTLASS/CuTe 代码看不懂 | 先看 CUTLASS 的 tutorial 文档，再看代码；必要时看 Reed Wanderman-Milne 的 CuTe 教程 |
| FlashAttention 源码太复杂 | 先只读 forward，忽略 backward；先读 FA1，再读 FA2 |
| Marlin kernel 看不懂 | 先理解简单的 W8A8 kernel，再看 W4A16 |
| 找不到开源贡献切入点 | 从 benchmark/test/docs 入手，不一定要改 kernel 代码 |
| 时间不够用 | 优先保证里程碑 1-3，里程碑 4-5 可以延后但不能跳过 |

---

## 七、面试差异化策略

38 岁在国内市场确实不利，但可以通过以下方式建立优势：

### 硬实力展示（简历 / GitHub）

1. **这个 repo 本身就是作品集** — 从 CUDA 基础到 CUTLASS 到 FlashAttention 的完整学习路径，有代码、有笔记、有性能数据
2. **开源 PR** — 一个被 vLLM/SGLang 合入的 PR 胜过一切空谈
3. **端到端优化案例** — 证明你不只是会写 kernel，还能定位和解决实际问题

### 差异化叙事（面试话术）

- **不是"转行新手"** → 而是"在后端 serving 领域有 N 年经验，现在深入到 GPU kernel 层做性能优化"
- **推荐系统经验** → 迁移到 LLM serving 的调度、batch 策略、延迟优化
- **年龄** → 转化为"工程成熟度"，不是"我什么都会"，而是"我知道怎么把一个系统从 80 分优化到 95 分"

### 目标公司 / 岗位方向

| 方向 | 适合度 | 说明 |
|------|--------|------|
| LLM 推理引擎团队（字节/DeepSeek/Moonshot） | ★★★★★ | 最对口，需要 CUTLASS + FA + 量化 |
| AI Infra / ML Platform（大厂） | ★★★★ | 后端经验 + GPU 优化能力的结合 |
| GPU 云服务（如 NVIDIA 生态公司） | ★★★★ | 推理优化 + serving 经验 |
| 端侧推理（手机/边缘设备） | ★★★ | 偏 TensorRT / 量化，与你方向有重叠 |

---

## 八、核心资源清单

### 必读源码

| 项目 | 重点 | GitHub |
|------|------|--------|
| CUTLASS 3.x | examples/, cute/tutorial/ | github.com/NVIDIA/cutlass |
| FlashAttention | csrc/flash_attn/ 下的 forward kernel | github.com/Dao-AILab/flash-attention |
| FlashInfer | decode attention kernel | github.com/flashinfer-ai/flashinfer |
| Marlin | marlin/marlin_cuda_kernel.cu | github.com/IST-DASLab/marlin |
| vLLM | vllm/attention/, vllm/model_executor/ | github.com/vllm-project/vllm |

### 必读论文

| 论文 | 重点章节 |
|------|---------|
| FlashAttention 1 (Dao 2022) | Algorithm 1, IO 复杂度分析 |
| FlashAttention 2 (Dao 2023) | parallelism 改进, forward algorithm |
| AWQ (Lin 2023) | Section 3: activation-aware quantization |
| Efficiently Scaling Transformer Inference (Pope 2022) | memory-bound 分析, KV cache |

### 学习资料

| 资料 | 用途 |
|------|------|
| CUTLASS 官方文档 (github wiki) | CUTLASS API 和架构理解 |
| Lei Mao 的博客 (leimao.github.io) | CUDA / CUTLASS / Tensor Core 系列文章 |
| Reed Wanderman-Milne 的 CuTe 教程 | CuTe 入门最好的资料 |
| 知乎 "BBuf" 系列 | CUDA 优化中文内容质量最高 |
| PyTorch Profiler + Nsight 联动 | 从 PyTorch 层找到需要优化的 kernel |

---

## 九、进度追踪

每完成一个里程碑，在下面打勾并记录日期：

- [ ] 里程碑 1：CUTLASS 定制 GEMM — 目标日期：____
- [ ] 里程碑 2：FlashAttention CUDA 源码 — 目标日期：____
- [ ] 里程碑 3：量化 Kernel — 目标日期：____
- [ ] 里程碑 4：开源贡献 PR — 目标日期：____
- [ ] 里程碑 5：端到端优化案例 — 目标日期：____
