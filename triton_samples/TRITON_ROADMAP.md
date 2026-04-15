# Triton 学习路线图

从 CUDA 开发者视角系统学习 OpenAI Triton，对标已有的 CUDA 学习内容。

Triton 是一种 Python-first 的 GPU 编程语言，核心理念：**用更少的代码写出接近手写 CUDA 的性能**。

---

## 环境准备

```bash
pip install triton torch
# Triton 依赖 PyTorch 作为张量后端
# 需要 NVIDIA GPU + CUDA driver（不需要单独安装 CUDA Toolkit）
```

验证安装：
```python
import triton
print(triton.__version__)
```

---

## CUDA vs Triton 核心概念映射

| CUDA 概念 | Triton 对应 | 说明 |
|-----------|------------|------|
| `__global__ void kernel()` | `@triton.jit` 装饰器 | kernel 定义方式 |
| `threadIdx.x`, `blockIdx.x` | `tl.program_id(axis)` | Triton 没有 thread 概念，以 **program** (≈ block) 为单位 |
| `blockDim.x * blockIdx.x + threadIdx.x` | `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` | 索引计算 |
| `__shared__` memory | 自动管理 | Triton 编译器自动决定 shared memory 使用 |
| `__syncthreads()` | 不需要 | 编译器自动插入同步 |
| shared memory tiling | `tl.load` + 循环 | Triton 自动做 tiling 和 prefetch |
| `float4` 向量化加载 | 自动 | 编译器自动选择最优访存宽度 |
| bank conflict | 自动避免 | 编译器自动处理 shared memory 布局 |
| Tensor Core (WMMA) | `tl.dot()` | 编译器自动使用 Tensor Core |
| cuBLAS | Triton kernel | 目标：用 Triton 写出 cuBLAS 级性能 |

**关键差异**：Triton 的编程粒度是 **block（program）**，不是 thread。你操作的是整个 block 的数据（向量/矩阵），编译器负责 thread 级调度。

---

## 学习路径

### Step 1: Vector Add — Hello Triton `01_vector_add.py`

**目标**：理解 Triton 最基本的编程模型

**核心知识点**：
- `@triton.jit` 装饰器定义 kernel
- `tl.program_id(0)` 获取 block 编号（等价于 CUDA 的 `blockIdx.x`）
- `tl.arange(0, BLOCK_SIZE)` 生成 block 内的偏移（等价于 `threadIdx.x`）
- `tl.load` / `tl.store` 读写 global memory
- `mask` 参数处理边界（等价于 CUDA 的 `if (idx < n)`）
- `grid` lambda 指定 launch 配置

**对标 CUDA**：`vector_add.cu`

**学习重点**：体会 Triton 如何把 thread-level 编程抽象为 block-level 编程

---

### Step 2: Fused Softmax `02_softmax.py`

**目标**：理解 kernel fusion 和 reduction 操作

**核心知识点**：
- **Kernel Fusion**：将 max、subtract、exp、sum、div 五个操作融合到一个 kernel
- `tl.max(x, axis=0)` — block 内归约
- `tl.exp()`, `tl.sum()` — 向量化数学操作
- 数值稳定性：`x - max(x)` 防止溢出
- 为什么 Triton 适合写 fused kernel（对比 PyTorch 的多个 kernel launch）

**性能对比**：
- PyTorch naive softmax（多次 kernel launch + 读写 global memory）
- Triton fused softmax（单次 launch，数据留在寄存器/shared memory）

**预期收益**：2-4x vs PyTorch naive（主要省掉了 memory traffic）

---

### Step 3: Matrix Multiplication `03_matmul.py`

**目标**：用 Triton 实现高性能 GEMM，对标你的整个 `matmul_optimize/` 系列

**核心知识点**：
- **2D grid launch**：`tl.program_id(0)` 和 `tl.program_id(1)` 分别对应 M 和 N 维度
- **分块累加**：沿 K 维度循环，每次加载一个 tile
- `tl.dot(a, b)` — Triton 会自动使用 Tensor Core（如果硬件支持）
- **L2 cache 优化**：通过 grouped ordering (swizzle) 提高 L2 命中率
- `tl.constexpr` 编译期常量

**Triton vs 手写 CUDA 对比**：

| 你的 CUDA 优化步骤 | Triton 中的实现 |
|-------------------|----------------|
| shared memory tiling | `tl.load` 循环（编译器自动用 shared memory） |
| 寄存器 tiling (TM×TN) | `BLOCK_SIZE_M/N/K` 参数控制 |
| bank conflict 处理 | 编译器自动 |
| double buffering | 编译器自动 prefetch |
| float4 向量化 | 编译器自动 |
| Tensor Core | `tl.dot()` 自动使用 |

**重点体会**：你在 CUDA 中花了 7 个 step 做的优化，Triton 用 ~50 行代码覆盖了大部分

---

### Step 4: Flash Attention `04_flash_attention.py`

**目标**：实现 Flash Attention — Triton 的杀手级应用场景

**核心知识点**：
- **Flash Attention 算法**：online softmax + tiling，避免存储完整的 N×N attention matrix
- 分块遍历 KV，在线更新 softmax 统计量（max 和 sum）
- `tl.where()` 条件操作实现 causal mask
- 多维指针运算和 block pointer

**为什么 Triton 适合 Flash Attention**：
- 算法涉及复杂的 fusion（QK^T → scale → mask → softmax → AV）
- 手写 CUDA 极其复杂（Flash Attention 原版 ~2000 行 CUDA）
- Triton 版本 ~100 行，性能接近

**对标**：PyTorch `F.scaled_dot_product_attention` 和 Flash Attention 2

---

### Step 5: Layer Normalization `05_layernorm.py`

**目标**：实现 LLM 推理中的关键算子，学习 forward + backward kernel

**核心知识点**：
- `tl.sum()`, `tl.sqrt()` 实现均值和方差计算
- Fused kernel：mean → variance → normalize → scale → bias 全部在一个 kernel 内
- Backward pass kernel：梯度计算的 Triton 实现
- 与 PyTorch `nn.LayerNorm` 性能对比

**LLM 关联**：
- LayerNorm / RMSNorm 是 Transformer 中频率最高的算子之一
- 推理时 batch size 小，这类 memory-bound 算子的 fusion 收益巨大

---

### Step 6: Quantized MatMul (INT8/FP8) `06_quantize_matmul.py`

**目标**：实现量化矩阵乘法，连接 LLM 推理优化

**核心知识点**：
- Weight-only quantization：权重 INT8，激活 FP16
- Per-channel / per-group scale factor
- `tl.dot()` 支持混合精度（INT8 输入 → FP32 累加）
- Dequantize 在 kernel 内 fused 完成

**LLM 关联**：
- 大模型推理的核心优化手段：4-bit/8-bit 量化
- 减少 memory bandwidth 需求（LLM 推理是 memory-bound）
- 对标 GPTQ、AWQ 等量化方案的底层 kernel

---

### Step 7: RMS Normalization `07_rmsnorm.py`

**目标**：实现 LLaMA/Mistral/Gemma 使用的 RMSNorm，对比 LayerNorm

**核心知识点**：
- RMSNorm 公式：`y = x / sqrt(mean(x²) + eps) * w`（无 mean subtraction, 无 bias）
- 比 LayerNorm 少一次 reduction，参数少一半
- Forward + Backward 完整实现（含 autograd）
- Backward 的数学推导：`dx = rrms * (dx_hat - x_hat * mean(dx_hat * x_hat))`

**LLM 关联**：
- 现代 LLM（LLaMA, Mistral, Gemma, Qwen）全部使用 RMSNorm 替代 LayerNorm
- 推理时是典型的 memory-bound 算子，fusion 收益显著

---

### Step 8: Paged Attention `08_paged_attention.py`

**目标**：理解 vLLM 的核心算子 — 分页 KV Cache 的 Attention 计算

**核心知识点**：
- **分页内存管理**：借鉴 OS 虚拟内存，KV Cache 按固定大小 page 分配
- **Page Table 间接寻址**：逻辑页 → 物理 block 的映射
- **Online Softmax + Paged KV 遍历**：逐页遍历，在线更新 softmax 统计量
- **GQA 支持**：多个 Q head 共享一个 KV head（Grouped Query Attention）
- **PagedKVCache 管理器**：分配、回收、碎片复用

**为什么 Paged Attention 重要**：
- 传统连续分配 max_seq_len → 内存浪费 50-90%
- 分页 → 按需分配 → 内存利用率接近 100% → 更大 batch → 更高吞吐
- vLLM, TensorRT-LLM, SGLang 等主流 serving 框架的基础

**对标**：vLLM PagedAttention, FlashInfer

---

## Auto-tuning：Triton 的独特优势

Triton 内置 auto-tuning 框架，自动搜索最优超参数：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        # ...更多配置
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

对比 CUDA：手动调参 → 重新编译 → 跑 benchmark → 循环。Triton 一行搞定。

---

## 性能记录表

| Kernel | 矩阵/序列长度 | Triton 时间 (ms) | PyTorch 时间 (ms) | cuBLAS 时间 (ms) | Triton/cuBLAS |
|--------|-------------|-----------------|------------------|-----------------|--------------|
| vector_add | N=100M | | | - | - |
| fused_softmax | 4096×4096 | | | - | |
| matmul | 2048×2048 | | | | |
| flash_attention | seq=2048, d=64 | | | - | |
| layernorm | 4096×4096 | | | - | |
| quantize_matmul | 4096×4096 | | | | |
| rmsnorm | 4096×4096 | | | - | |
| paged_attention | seq=4096, d=128 | | | - | |

---

## 推荐学习资源

- Triton 官方教程：https://triton-lang.org/main/getting-started/tutorials/
- Triton GitHub：https://github.com/triton-lang/triton
- 论文：*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*
- Flash Attention 论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Unsloth Triton kernels（实战参考）：https://github.com/unslothai/unsloth
