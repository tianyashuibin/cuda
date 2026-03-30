# 算子融合（Operator Fusion）

## 定义

把多个 kernel 合并成一个 kernel，同时消除中间结果写回显存的开销。

核心收益两点：
1. **多次 kernel launch → 一次**：消除 CPU 反复调度 GPU 的开销
2. **多次显存读写 → 一次**：中间结果留在寄存器或 shared memory，不落地显存

其中第 2 点通常是更大的收益，因为显存（HBM）带宽有限，反复读写大矩阵的代价远高于 launch 调度开销。

---

## 原理

以 `ReLU → Dropout → LayerNorm` 为例：

**未融合（3 个 kernel）：**
```
显存读 x → [ReLU]     → 显存写 y
显存读 y → [Dropout]  → 显存写 z
显存读 z → [LayerNorm]→ 显存写 out

3 次读显存，3 次写显存
```

**融合后（1 个 kernel）：**
```
显存读 x → [ReLU → Dropout → LayerNorm] → 显存写 out

1 次读显存，1 次写显存，中间结果全在寄存器里
```

**融合后的 kernel 代码形态：**
```cpp
__global__ void FusedKernel(float *x, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = x[i];        // 读一次显存
    val = max(val, 0.0f);    // ReLU，val 在寄存器
    val = dropout(val);      // Dropout，val 在寄存器
    val = layernorm(val);    // LayerNorm，val 在寄存器
    out[i] = val;            // 写一次显存
}
```

**为什么 element-wise 算子可以这样做？**

每个元素的计算完全独立，线程 i 只处理元素 i，不需要等其他线程的结果，中间值可以直接存在寄存器里向下传递。

---

## 数据流对比

```
未融合：
  显存读 → 计算 → 显存写 → 显存读 → 计算 → 显存写 → 显存读 → 计算 → 显存写

融合后：
  显存读 → 计算 → 计算 → 计算 → 显存写
```

---

## 哪些算子可以融合

| 可以融合 | 不容易融合 |
|---------|-----------|
| Element-wise（逐元素） | Reduction（需要跨线程 __syncthreads）|
| 相邻的 pointwise 算子 | MatMul（数据依赖复杂）|
| 激活 + 加法 + 归一化 | Conv（访问模式复杂）|

Reduction 类算子（softmax、layernorm）需要跨线程通信，融合难度更高，但 Flash Attention 证明了精心设计的算法可以做到。

---

## 实际案例

### Flash Attention
朴素 Attention 有 6~8 个 kernel，中间的 N×N score 矩阵（序列长度 4096 时达 64MB）需要反复读写显存。

Flash Attention 将其融合为 1 个 kernel，把中间结果保存在 shared memory 里：

```
朴素：S = Q@K^T → 写显存 → softmax → 写显存 → S@V → 写显存
Flash：分块加载 Q/K/V 到 shared memory，在 shared memory 内完成全部计算，只写一次显存
```

主要加速来源是消除 O(N²) 的显存读写，而不仅仅是减少 launch 次数。

---

## PyTorch 中的融合机制

| 机制 | 说明 |
|------|------|
| `torch.compile` | 编译时分析计算图，自动融合 element-wise 算子，生成 Triton kernel |
| cuDNN / cuBLAS 内置融合 | `addmm`（matmul + add）等底层已融合 |
| 手写 CUDA kernel | Flash Attention 等性能关键路径手动实现 |
| CUDA Graph | 不减少 kernel 数量，但消除所有 launch 的 CPU 调度开销 |

### torch.compile 示例
```python
@torch.compile
def forward(x):
    x = F.relu(x)    # ┐
    x = x * 0.5      # ├─ 自动融合为一个 Triton kernel
    x = x + 1.0      # ┘
    return x
```

---

## CPU 上的算子融合

算子融合是通用优化思路，CPU 上同样有效，原理完全一致，只是存储层次的名字不同。

**CPU 存储层次（从快到慢）：**

```
寄存器      ~1 cycle
L1 cache   ~4 cycle    (per core, 32~64 KB)
L2 cache   ~12 cycle   (per core, 256 KB~1 MB)
L3 cache   ~40 cycle   (shared, 8~64 MB)
DRAM       ~200 cycle  ← 未融合时中间结果落这里
```

未融合时，每个算子执行完把结果写回 DRAM，下一个算子再从 DRAM 读取，跨越 200 cycle 的延迟。融合后，中间结果留在 L1/L2 cache 甚至寄存器里，只需 1~12 cycle。

**CPU 与 GPU 的对比：**

```
GPU：中间结果留在 寄存器 / shared memory，避免写回 HBM（显存，~100ns）
CPU：中间结果留在 寄存器 / L1/L2 cache，避免写回 DRAM（内存，~60ns）
```

**本质是同一个思路：**

> 让数据尽量待在离计算单元近的快速存储里，减少对慢速存储的读写次数。

CPU 上没有 kernel launch 的概念，所以融合的收益只来自缓存效果（减少 DRAM 读写），而 GPU 上融合同时带来两个收益：减少显存读写 + 减少 kernel launch 次数。

---

## 与 CUDA Graph 的区别

| | 算子融合 | CUDA Graph |
|--|---------|-----------|
| 解决的问题 | 显存读写次数过多 + launch 次数过多 | CPU launch overhead |
| 手段 | 合并 kernel，中间结果留在寄存器/shared memory | 录制图，一次提交所有 launch |
| 主要收益 | 减少显存带宽消耗 | 减少 CPU 调度延迟 |
| 典型场景 | 训练/推理的计算密集路径 | 推理的重复 forward pass |
