# GEMM 优化路线图

基于 `matmul_sharemem.cu`（shared memory tiling）逐步优化，对标生产级 GEMM（cuBLAS/CUTLASS）。

每一步完成后记录性能数据，做对比分析。

## 基准：当前实现 (matmul_sharemem.cu)

- Shared memory tiling，BLOCK_SIZE=16
- 每个线程计算 C 的 1 个元素
- 矩阵尺寸必须是 BLOCK_SIZE 的整数倍

---

## 优化步骤

### Step 1: 增大 Tile 尺寸 (BLOCK_SIZE 32)

**目标**：减少 global memory 访问次数，提高数据复用率。

**原理**：
- BLOCK_SIZE 从 16 增大到 32，global memory 读取减少为原来的 1/2
- 每次 tile 加载更多数据到 shared memory，复用次数从 16 → 32

**注意事项**：
- shared memory 用量从 2×16×16×4 = 2KB → 2×32×32×4 = 8KB
- 每个 block 线程数从 256 → 1024（接近硬件上限）
- 可能影响 occupancy

**预期收益**：10-20%

---

### Step 2: 寄存器 Tiling (Thread Tile)

**目标**：每个线程计算多个 C 元素（如 4×4 或 8×8），减少 shared memory 访问。

**原理**：
- 从 shared memory 加载到寄存器后，一个值可被复用多次
- 比如线程负责 C 的 4×4 子块，A 的一行值可复用 4 次，B 的一列值可复用 4 次
- 访存比（计算/访存）从 2:1 提升到 ~8:1

**实现要点**：
```
Thread Tile = TM × TN (如 8×8)
Block Tile = BM × BN (如 128×128)
每个 block 的线程数 = (BM/TM) × (BN/TN) = 16×16 = 256
每个线程用 TM×TN = 64 个寄存器存 C 的累加值
```

**预期收益**：2-4x（这是最关键的一步优化）

---

### Step 3: Double Buffering (预取/流水线)

**目标**：用计算掩盖访存延迟，让 shared memory 加载和计算重叠执行。

**原理**：
- 分配两份 shared memory buffer (A/B 各两份)
- 当线程在计算当前 tile 时，同时预取下一个 tile 到另一份 buffer
- 通过 `__syncthreads()` 切换 buffer，实现流水线

**实现要点**：
```
As[2][BM][BK]  // 双份 buffer
Bs[2][BK][BN]
循环：
  加载 tile[m+1] 到 buffer[(m+1)%2]  ← 异步
  计算 tile[m] 从 buffer[m%2]         ← 同时执行
  __syncthreads()
```

**预期收益**：10-30%

---

### Step 4: 向量化访存 (Vectorized Load/Store)

**目标**：使用 float4 一次加载 128bit，提高 global memory 带宽利用率。

**原理**：
- 单次 `float` 加载 = 32bit 事务
- 单次 `float4` 加载 = 128bit 事务，充分利用内存总线宽度
- 减少访存指令数量

**实现要点**：
```cpp
// 替换
float val = A[offset];
// 为
float4 val = reinterpret_cast<float4*>(&A[offset])[0];
```

**注意事项**：
- 要求地址 16 字节对齐
- 矩阵宽度需要是 4 的倍数

**预期收益**：10-20%

---

### Step 5: Warp 级优化

**目标**：利用 warp 内线程的隐式同步和 shuffle 指令。

**原理**：
- 同一 warp 的 32 个线程天然同步，不需要 `__syncthreads()`
- 可用 `__shfl_sync` 在 warp 内交换数据，避免 shared memory 中转
- 合理安排 warp 对 tile 的映射，避免 bank conflict

**实现要点**：
- Warp tile：每个 warp 负责一块连续的 C 子区域
- 安排 shared memory 布局避免 bank conflict（如 padding 或 swizzle）

**预期收益**：5-15%

---

### Step 6: 支持任意矩阵尺寸

**目标**：移除矩阵尺寸必须是 BLOCK_SIZE 整数倍的限制。

**实现要点**：
- 边界 tile 需要做 bounds checking
- 越界位置填 0（不影响乘法结果）
```cpp
if (row < M && col < K)
    As[ty][tx] = A[row * K + col];
else
    As[ty][tx] = 0.0f;
```

**预期收益**：无性能提升，但功能完整性必须有

---

### Step 7: Tensor Core (WMMA)

**目标**：使用 Tensor Core 硬件加速矩阵乘法。

**原理**：
- Tensor Core 每周期可完成 16×16×16 的矩阵乘加（FP16 输入 → FP32 累加）
- 单条 WMMA 指令 = 手写 kernel 数百条 FMA 指令

**实现要点**：
```cpp
#include <mma.h>
using namespace nvcuda::wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
load_matrix_sync(a_frag, ...);
load_matrix_sync(b_frag, ...);
mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**注意事项**：
- 需要 Volta 架构及以上 (SM >= 70)
- 输入通常为 FP16，需要处理精度问题

**预期收益**：2-4x（相比纯 FP32 CUDA Core）

---

## 性能记录表

| 版本 | 矩阵尺寸 | Kernel 时间 (ms) | GFLOPS | 相比基准加速比 | 达到 cuBLAS 百分比 |
|------|----------|-----------------|--------|-------------|-----------------|
| baseline (shared mem 16) | 1024 | - | - | - | |
| step1: tile 32 | 1024 | 5.508 | 389.8 | 1.0x (基准) | |
| step2: 寄存器 tiling | 1024 | 1.259 | 1705.1 | 4.37x | |
| step2.5: bank conflict fix (PAD+swizzle) | 2048 | 5.713 | 3007.2 | 1.47x vs step2 | 55.2% |
| step3+4: double buffer + float4 store | 2048 | 6.050 | 2839.9 | 1.03x vs step2.5 | 52.1% |
| step5: warp tiling (未修复) | 2048 | 9.713 | 1768.8 | 0.98x vs v4 (退步) | 32.5% |
| step6: 任意尺寸 | 2048 | | | | |
| step7: tensor core (naive WMMA) | 2048 | 10.413 | 1649.9 | | 30.3% |
| cuBLAS (参考) | 2048 | 3.154 | 5447.1 | | 100% |

> GFLOPS 计算公式：2 × M × N × K / (kernel_time_sec) / 1e9

---

## 参考资料

- CUTLASS: https://github.com/NVIDIA/cutlass
- CUDA C++ Programming Guide: Matrix Multiply
- 知乎/博客：手写高性能 GEMM 系列