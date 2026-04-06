# Register Tiling GEMM — Profiling 分析报告

**Kernel**: `MatMulRegTilingKernel`
**矩阵尺寸**: 1024x1024 / 2048x2048
**GPU**: Tesla T4 (SM 7.5, 40 SMs, 65536 regs/SM, 320 GB/s DRAM)
**参数**: BM=128, BN=128, BK=8, TM=8, TN=8, Block=16x16=256 threads

---

## 一、Nsight Systems (nsys) — 系统级全局视角

nsys 回答的问题是："时间花在哪了？"

### 1. CUDA API Summary — CPU 端耗时

| API 调用 | 耗时 | 占比 | 解读 |
|---------|------|------|------|
| `cudaMalloc` | 188.9 ms | 94.7% | 首次调用触发 CUDA context 初始化，一次性开销，不是真正瓶颈 |
| `cudaMemcpy` | 5.0 ms | 2.5% | H2D + D2H 数据传输 |
| `cudaLaunchKernel` | 1.6 ms | 0.8% | 两次 launch，第一次 1.62ms（冷启动），第二次 15us（热启动） |
| `cudaEventSynchronize` | 1.45 ms | 0.7% | 等 kernel 完成，约等于 kernel 实际执行时间 |

**结论**：CPU 端没有瓶颈，cudaMalloc 的大耗时是一次性开销。

### 2. GPU Kernel Summary

| 指标 | 值 | 解读 |
|------|-----|------|
| Instances | 2 | warmup + 正式各一次 |
| Avg | 1.445 ms | 两次几乎一样，warmup 有效 |
| StdDev | 1,878 ns | 极小，执行非常稳定 |

### 3. Memory Transfer — H2D/D2H

| 操作 | 大小 | 耗时 | 带宽 |
|------|------|------|------|
| H2D | 8.39 MB (A + B) | 1.60 ms | ~5.2 GB/s |
| D2H | 4.19 MB (C) | 1.60 ms | ~2.6 GB/s |

传输不是重点，kernel 执行时间才是优化目标。

### nsys 总结

> 宏观上没有问题：没有 GPU idle gap，没有多余的同步，kernel 就是唯一瓶颈。

---

## 二、Nsight Compute (ncu) — Kernel 深度分析

以下用 **2048x2048** 的数据，SM 利用更充分（3.2 waves），数据更有代表性。

### 1. Speed of Light — 整体效率

| 指标 | 值 | 解读 |
|------|-----|------|
| Compute (SM) Throughput | **40.68%** | 计算管线利用不到一半 |
| Memory Throughput | **43.04%** | 访存管线也不到一半 |
| FP32 Peak | **38%** | 达到 T4 峰值 8.1 TFLOPS 的 1/3 |
| DRAM Throughput | 4.61% | DRAM 几乎没压力 |
| L1/TEX Cache Throughput | **86.07%** | L1 非常繁忙（shared mem 走 L1 通道）|

**关键判断**：Compute 和 Memory 都 <60% → **latency bound（延迟受限）**。
warp 在等待某些东西，导致管线空转。

### 2. Occupancy — 占用率

| 指标 | 值 | 解读 |
|------|-----|------|
| Registers Per Thread | **96** | occupancy 的限制因素 |
| Block Size | 256 (16x16) | 8 warps/block |
| Block Limit Registers | **2 blocks/SM** | 96 reg x 256 threads = 24576，T4 有 65536 reg/SM，只够放 2 个 block |
| Block Limit Shared Mem | 4 blocks/SM | 8.19KB/block，不是瓶颈 |
| Theoretical Occupancy | **50%** | 16 warps / 32 max warps per SM |
| Achieved Occupancy | **47.97%** | 接近理论值 |

50% occupancy 对 register tiling 来说可接受——用低 occupancy 换取更少的 shared memory 访问。

### 3. Scheduler Statistics — 调度器效率（核心问题）

| 指标 | 值 | 解读 |
|------|-----|------|
| Active Warps Per Scheduler | 3.82 / 8 max | 因为 50% occupancy |
| Eligible Warps Per Scheduler | **0.42** | 只有 0.42 个 warp 随时可以发射指令 |
| No Eligible | **68.71%** | 近 70% 的周期，调度器找不到任何可执行的 warp |
| Issued Warp Per Scheduler | 0.31 | 每 ~3.2 个周期才发射一条指令 |

**最严重的问题**：3.82 个 active warps 里平均只有 0.42 个 eligible。大部分 warp 都在 stall。

### 4. Warp State Statistics — warp 在等什么？

**主要 stall 原因：MIO Throttle (Short Scoreboard) — 占 57.6%**

- 每条指令平均等待 **12.22 cycles**
- 其中 **7.0 cycles** 花在等待 MIO（Memory I/O）操作
- ncu 明确说：**主要原因是 shared memory 操作**

warp 发出 shared memory load 指令后，后续指令依赖结果必须等待。

### 5. Memory Workload — 具体访存问题

#### 问题 A：Shared Memory Bank Conflict（最大的问题）

| 指标 | 值 |
|------|-----|
| 平均 bank conflict | **3.2-way** |
| Shared load 总请求 | 41,943,040 |
| 产生的 bank conflict | **67,108,864** |
| 冲突占总 wavefronts | **50%** |
| ncu 估计加速潜力 | **43.7%** |

**原因分析**：
```c
regA[tm] = As[ty * TM + tm][bk];  // As[BM][BK], BK=8
```
同一 warp 内线程的 `ty` 相同，`tm` 从 0~7 遍历。访问 `As[ty*8+0..7][bk]`，即连续 8 行的同一列。
`As` 列数是 BK=8，行间距 8 floats = 32 bytes。Shared memory 有 32 个 bank，每 bank 4 bytes，
32 bytes 间距意味着每隔 8 个 bank 回到同一 bank → **4-way bank conflict**（实测 3.2-way 是混合效果）。

#### 问题 B：Global Store Uncoalesced

| 指标 | 值 |
|------|-----|
| 每 sector 有效利用 | **4.0 / 32 bytes** (仅 12.5%) |
| Excessive sectors | 3,670,016 (18%) |
| ncu 估计加速潜力 | **10%** |

**原因分析**：写回 C 矩阵时：
```c
C[globalRow * N + globalCol] = regC[tm][tn];
```
同一 warp 的线程沿 `tx` 排列，`tx` 对应的列间隔 TN=8 个元素，导致非连续写入。

### 6. Compute Workload — 计算管线

| 指标 | 值 | 解读 |
|------|-----|------|
| IPC Active | 1.25 | 每周期执行 1.25 条指令 |
| Issue Slots Busy | 31.34% | 只有 1/3 的 slot 在工作 |
| SM Busy | 45.14% | SM 不到一半时间在忙 |
| FMA 管线利用率 | 45.1% | 最高的管线，not bottleneck |

计算本身没问题，FMA 是最繁忙管线（符合预期），但 warp stall 导致喂不满。

---

## 三、Bank Conflict 深入分析

### 3.2-way 的来源拆解

ncu 报告的 3.2-way 是 As 和 Bs 两个 shared memory 的 **加权平均**：

#### As 的 bank conflict (2-way)

```c
regA[tm] = As[ty * TM + tm][bk];  // As[BM][BK] = As[128][8]
```

block = (16,16)，warp 内包含 2 个连续 ty 值（各 16 个 tx 线程）。
同 ty 的 16 个 tx 线程读同一地址 → broadcast，无 conflict。
两个 ty 值（间隔 TM=8 行）读不同地址，检查 bank：

```
bank 间距 = (TM × BK) % 32 = (8 × 8) % 32 = 64 % 32 = 0 → 同一 bank → 2-way conflict
```

#### Bs 的 bank conflict (4-way)

```c
regB[tn] = Bs[bk][tx * TN + tn];  // Bs[BK][BN] = Bs[8][128]
```

同一 warp 的 16 个 tx 值访问列 `tx*8+tn`，列间距 8 floats = 32 bytes = 完整 bank 周期：

```
tx=0  → bank = (0*8+tn)  % 32 = tn
tx=4  → bank = (4*8+tn)  % 32 = tn      ← 和 tx=0 相同
tx=8  → bank = (8*8+tn)  % 32 = tn      ← 和 tx=0 相同
tx=12 → bank = (12*8+tn) % 32 = tn      ← 和 tx=0 相同
→ {tx=0,4,8,12} 映射到同一 bank → 4-way conflict
```

As (2-way) + Bs (4-way) 平均 ≈ **3.2-way**，和 ncu 报告完全吻合。

---

## 四、PAD=4 修复实验 — 失败分析

### 实验结果 (2048x2048, T4)

| 指标 | v1 (原版) | v2 (PAD=4) | 变化 |
|------|-----------|------------|------|
| Kernel 时间 | 8.350 ms | 8.465 ms | 更慢了 |
| GFLOPS | 2057.6 | 2029.5 | -1.4% |
| Bank conflict | 3.2-way, 50% | 3.2-way, 50% | 无变化 |
| MIO stall | 7.0 cycles (57.6%) | 7.2 cycles (58.6%) | 无变化 |
| Shared mem/block | 8.19 KB | 10.24 KB | +2 KB（浪费） |
| Shared store conflict | 无 | 1.5-way (新增) | 更差了 |

### 为什么 PAD=4 完全无效

As 的 bank 间距公式：`(TM × stride) % 32`，stride = BK + PAD

```
PAD=4: stride=12, 8 × 12 = 96,  96 % 32 = 0  → 同一 bank（无效！）
PAD=1: stride=9,  8 × 9  = 72,  72 % 32 = 8  → 不同 bank ✓
PAD=2: stride=10, 8 × 10 = 80,  80 % 32 = 16 → 不同 bank ✓
PAD=3: stride=11, 8 × 11 = 88,  88 % 32 = 24 → 不同 bank ✓
```

**规律**：PAD % 4 == 0 时完全无效。PAD=4 恰好是最差的选择之一。

### Bs 的 bank conflict 无法用 padding 修复

Bs 的 conflict 来自**行内**访问模式（tx * TN 间距），padding 只影响行间距。
要修复 Bs 需要：
- **XOR swizzle**：`Bs[bk][col ^ (col >> 2)]` 等位运算打散 bank 映射
- **转置存储**：改变 Bs 的存储布局使 warp 访问连续 bank
- **调整 warp-to-tile 映射**：改变线程分配方式

### 关键教训

1. Padding 值必须经过 bank 公式验证，不能随意选取
2. 要分别分析 As 和 Bs 的 conflict，它们的成因不同
3. 行内访问模式的 conflict（如 Bs）padding 无法解决，需要 swizzle 等技术

---

## 五、v3 (PAD+Swizzle) 修复效果

v3 = As PAD=1 + Bs XOR swizzle `col ^ ((col >> 3) << 1)`

| 指标 | v1 (原版) | v3 (修复后) | 变化 |
|------|-----------|------------|------|
| Kernel 时间 (2048) | 8.350 ms | 5.713 ms | **-31.6%** |
| GFLOPS | 2057.6 | 3007.2 | **+46.2%** |
| Bank conflict | 3.2-way, 50% | ~0-way, 3% | **基本消除** |
| IPC Active | 1.25 | 1.92 | +53.6% |

---

## 六、v4 (Double Buffering + Float4 Store) 分析

### 1. 性能数据（正常运行，无 profiler 开销）

| 指标 | v3 (PAD+swizzle) | v4 (double buf + float4) | 变化 |
|------|---|---|---|
| Kernel 时间 (2048) | 6.242 ms | 6.050 ms | -3.1% |
| GFLOPS | 2752.2 | 2839.9 | +3.2% |
| nsys Avg (含 profiler) | 9.372 ms | 9.091 ms | -3.0% |

提升幅度不大，仅 **1.03x**。

### 2. ncu 关键指标（与 v1 原版对比）

| 指标 | v1 (原版) | v4 (最新) | 变化 |
|------|-----------|-----------|------|
| **Registers/Thread** | 96 | **128** | +33%（double buffer 引入额外状态）|
| **IPC Active** | 1.25 | **2.04** | +63% |
| **SM Busy** | 45.14% | **69.68%** | +24.5pp |
| **Issue Slots Busy** | 31.34% | **50.97%** | +19.6pp |
| Compute Throughput | 40.68% | **66.06%** | +25.4pp |
| Memory Throughput | 43.04% | **66.06%** | +23.0pp |
| **Eligible Warps/Scheduler** | 0.42 | **1.17** | 近 3x |
| **No Eligible** | 68.71% | **49.30%** | -19.4pp |
| Warp Cycles/Instruction | 12.22 | **7.31** | -40.2% |
| FMA Pipeline | 45.1% | **69.7%** | +24.6pp |
| L1/TEX Cache Throughput | 86.07% | **74.77%** | -11.3pp |
| DRAM Throughput | 4.61% | **7.02%** | +2.4pp |
| Occupancy (理论) | 50% | **50%** | 不变 |
| Occupancy (实测) | 47.97% | **46.57%** | 略降 |

**ncu 判断变化**：v1 是 "latency bound"（延迟受限），v4 变成 **"Compute and Memory are well-balanced"**（计算与访存均衡）。

### 3. v4 新引入的问题

#### 问题 A：寄存器压力到达极限 (96 → 128 regs/thread)

Double buffering 引入两套 buffer 指针和额外控制变量，编译器需要更多寄存器：
- 128 regs × 256 threads = 32,768 regs/block
- T4 每 SM 65,536 regs → 正好放 2 blocks/SM（和 v3 一样）
- **已在极限边缘**：再多 1 个 register 就只能放 1 block，occupancy 从 50% 暴跌到 25%

#### 问题 B：Shared store 新增 1.8-way bank conflict

| 指标 | 值 |
|------|-----|
| Shared store 请求 | 4,194,304 |
| Bank conflict wavefronts | 3,218,441 (43.36%) |
| ncu 估计加速潜力 | 43.36% |

原因：double buffer 的协作加载阶段写入 `Bs[buf][][]` 时，store 模式与 read 模式不同，产生新的 bank conflict。

#### 问题 C：Global store 改善但未达最优 (12.5% → 50%)

| 指标 | v1 | v4 |
|------|-----|-----|
| 每 sector 有效利用 | 4.0/32 bytes (12.5%) | 16.0/32 bytes (50%) |
| Excessive sectors | 18% | 3% |
| ncu 估计加速潜力 | 10% | 1.9% |

float4 将单次 store 从 32bit 提升到 128bit（16 bytes），但一个 sector 是 32 bytes。
原因：TN=8 导致相邻 tx 线程写入间距 = 8 floats = 32 bytes = 整个 sector 宽度，每个 float4 只填半个 sector。

### 4. 为什么只提升了 3%

**核心原因：double buffering 的收益被寄存器增长和新 bank conflict 部分抵消。**

- 好处：计算和访存重叠执行，`__syncthreads()` 从 2 次/tile 减少到 1 次
- 代价：寄存器 96→128，新增 shared store 1.8-way bank conflict

v3 的 bank conflict 修复已经把 warp stall 从 57.6% 大幅降低，double buffering 本来要解决的 latency hiding 问题在 v3 之后已经不那么严重，边际收益有限。

---

## 七、当前问题总结和下一步方向

### 已完成优化

| 步骤 | 优化 | 效果 |
|------|------|------|
| v1→v3 | As PAD=1 + Bs XOR swizzle | bank conflict 50%→3%，GFLOPS +46% |
| v3→v4 | Double buffering + float4 store | IPC 1.92→2.04，global store 12.5%→50% |

### 累积优化效果（v1 → v4）

| 指标 | v1 (原版 2048) | v4 (最新 2048) | 累积提升 |
|------|------|------|------|
| Kernel 时间 | 8.350 ms | 6.050 ms | **-27.5%** |
| GFLOPS | 2057.6 | 2839.9 | **+38.0%** |
| IPC | 1.25 | 2.04 | **+63.2%** |
| SM Busy | 45.14% | 69.68% | **+24.5pp** |
| ncu 状态 | latency bound | **balanced** | 质变 |

### 剩余问题（按优先级）

| 优先级 | 问题 | 影响 | 可能的修复方案 |
|--------|------|------|---------------|
| **P0** | 寄存器 128/thread（极限边缘）| 任何增加都导致 occupancy 暴跌 | 用 `__launch_bounds__` 限制，或减少 double buffer 状态 |
| **P1** | Shared store 1.8-way conflict | 43% wavefronts 冲突 | 优化 double buffer 的 store 模式 |
| **P2** | Global store 50% 效率 | sector 利用仅一半 | 改变 warp-to-tile 映射（warp 级优化）|
| **P3** | Occupancy 50% | 只有 2 blocks/SM | 寄存器限制，目前可接受 |

### 待研究方向

1. **Warp 级优化** (Step 5) — 重新设计 warp 到 tile 的映射，改善 store coalescing 和 bank conflict
2. **任意矩阵尺寸** (Step 6) — 添加边界检查
3. **Tensor Core WMMA** (Step 7) — 使用硬件矩阵乘加指令

---

## 八、v5 (Warp Tiling) 分析 — 寄存器膨胀导致 Occupancy 崩塌

### 1. 设计思路

引入显式 **三级 Tile 层次**：Block (128x128) → Warp (32x64) → Thread (8x8)

使用 1D block (256 threads)，通过 warpId/laneId 显式映射：
```
warpId = tid / 32          (0..7)
warpRow = warpId / 2       (0..3, M 方向)
warpCol = warpId % 2       (0..1, N 方向)
threadRowInWarp = laneId / 8  (0..3)
threadColInWarp = laneId % 8  (0..7)
```

**理论优势**：warp 内多个 lane 读同一 shared memory 地址 → 硬件广播，减少 bank conflict 和总访存量。

### 2. 性能结果（2048x2048, T4）

| 指标 | v4 (dbuf+f4st) | v5 (warp tile) | 变化 |
|------|----------------|----------------|------|
| Kernel 时间 | 9.523 ms | **9.713 ms** | **+2% 更慢** |
| GFLOPS | 1804.0 | **1768.8** | -2.0% |
| Speedup v5/v4 | - | **0.98x** | 退步 |
| Max diff | - | 0.000000e+00 | 正确性 PASSED |

### 3. ncu 关键指标

| 指标 | v4 | v5 (warp tile) | 变化 |
|------|-----|----------------|------|
| **Registers/Thread** | 128 | **144** | **+16 (根因!)** |
| **Occupancy (理论)** | 50% | **25%** | **腰斩** |
| **Occupancy (实测)** | ~46% | **24.96%** | 腰斩 |
| Block Limit Registers | 2 blocks/SM | **1 block/SM** | 瓶颈 |
| Block Limit Shared Mem | 3 blocks/SM | 3 blocks/SM | 不变 |
| Eligible Warps/Scheduler | ~1.17 | **0.72** | -38.5% |
| No Eligible | ~49% | **58.10%** | +9pp |
| IPC Active | ~2.04 | **1.68** | -17.6% |
| Compute Throughput | ~66% | **58.75%** | -7.3pp |
| Memory Throughput | ~66% | **44.23%** | -21.8pp |
| Warp Cycles/Inst | ~7.31 | **4.76** | 改善 |
| SM Busy | ~70% | **64.35%** | -5.7pp |
| FMA Pipeline | ~70% | **64.3%** | -5.7pp |
| Shared store conflict | 1.8-way | **1.5-way** | 略改善 |
| Global store util | 16/32 (50%) | **16/32 (50%)** | 不变 |
| Shared excessive wavefronts | - | 4% | 低 |
| Global excessive sectors | - | 3% | 低 |

### 4. 根因分析：寄存器膨胀 → Occupancy 崩塌

**直接原因**：
```
v4: 128 regs/thread × 256 threads = 32,768 regs/block → 65,536 / 32,768 = 2 blocks/SM → 50%
v5: 144 regs/thread × 256 threads = 36,864 regs/block → 65,536 / 36,864 = 1 block/SM → 25%
```

**为什么多了 16 个寄存器**：

warp tiling 引入更复杂的索引计算（6 个新变量: `warpId, laneId, warpRow, warpCol, threadRowInWarp, threadColInWarp`），加上 shared memory 寻址中的乘法（`warpRow * WM`, `warpCol * WN` 等），编译器无法完全优化掉这些中间值，导致寄存器从 128 膨胀到 144。

**后果链**：
```
+16 registers → 1 block/SM (vs 2) → occupancy 25% (vs 50%)
  → active warps 从 ~4/scheduler 降到 ~2/scheduler
  → eligible warps 从 1.17 降到 0.72
  → 58% 周期无指令可发射
  → 吞吐量下降
```

### 5. warp tiling 的理论优势是否实现

| 预期优势 | 是否实现 | 说明 |
|----------|----------|------|
| As 读广播 (0 conflict) | 部分 | PAD 已经在 v3/v4 解决了 As conflict |
| Bs 读广播 (0 conflict) | 部分 | swizzle 已在 v3/v4 大幅缓解 |
| 更少唯一地址 (-33%) | 是 | 但被 occupancy 腰斩完全抵消 |
| 紧凑 warp tile (32x64) | 是 | L2 局部性改善，但效果有限 |
| Warp Cycles/Inst 改善 | 是 | 4.76 vs 7.31，每条指令延迟确实降了 |

**核心教训**：算法层面的优化（广播、减少地址数）被硬件资源限制（寄存器上限）完全抵消。在 T4 这种 65536 regs/SM 的硬件上，128 regs/thread 是一个关键分界线。

### 6. 修复方案（未实施）

#### 方案 A：`__launch_bounds__` 限制寄存器（推荐）
```cuda
__global__ __launch_bounds__(256, 2)  // 256 threads/block, 至少 2 blocks/SM
void MatMulWarpTilingKernel_v5(...)
```
- 告诉编译器必须将寄存器压到 ≤128，恢复 50% occupancy
- 代价：可能产生 register spill 到 local memory
- 通常 occupancy 的收益 > spill 代价

#### 方案 B：编译器 flag 限制寄存器
```
nvcc -maxrregcount=128 ...
```
- 全局限制，不如 `__launch_bounds__` 精确

#### 方案 C：简化索引计算
- 预计算 `warpRow * WM` 等偏移为单个变量
- 减少 inner loop 中的活跃变量数
- 需要仔细分析编译器生成的 SASS 代码

#### 方案 D：调整 Tile 参数减少寄存器
- 减小 TM/TN（如 4x8），减少 `regC` 数组大小
- 代价：降低计算/访存比

### 7. 结论

v5 的 warp tiling 设计在**算法层面是正确的**（广播减少 bank conflict、紧凑 warp tile 改善局部性），但在 T4 上被**寄存器压力**完全抵消。核心教训：**CUDA 优化必须同时考虑算法改进和硬件资源约束**，否则理论收益会被实际的 occupancy 下降吞掉。

修复方向明确：加 `__launch_bounds__(256, 2)` 即可恢复 occupancy，预期能让 warp tiling 的理论优势真正体现出来。

---

## 九、Tensor Core 与 cuBLAS 参考数据

### 测试环境

GPU: Tesla T4 (SM 7.5), 矩阵尺寸: 2048x2048

### 结果

| 版本 | Kernel 时间 (ms) | GFLOPS | 说明 |
|------|-----------------|--------|------|
| Tensor Core (naive WMMA) | 10.413 | 1649.9 | 每 block 1 个 warp, 无 shared memory tiling, 无 double buffer |
| cuBLAS (FP32 sgemm) | 5.776 | 2974.4 | NVIDIA 官方优化, 参考基准 100% |

### Tensor Core 为什么比手写 FP32 kernel 慢

naive WMMA 实现（`matmul_tensorcore.cu`）非常简单：
- 每个 block 只有 1 个 warp (32 threads)，计算一个 16x16 tile
- 直接从 global memory `load_matrix_sync`，没有 shared memory 缓存
- 没有 tiling / double buffering / 向量化访存

这意味着：
1. **极低的数据复用**：每个 16x16 tile 的 A/B 数据从 global memory 读取后只用一次
2. **极低的 occupancy**：每 block 1 warp = 每 SM 最多 32 warps，但实际受限于 grid 大小
3. **Tensor Core 本身很快**，但被 global memory latency 完全掩盖

优化后的 Tensor Core（如 CUTLASS）会加入 shared memory tiling + double buffer + warp 级流水线，可达到 10+ TFLOPS。

### cuBLAS vs 手写 kernel

| 版本 | GFLOPS | 达到 cuBLAS % |
|------|--------|--------------|
| v1 原版 reg tiling | ~2050 | 37.6% |
| v3 PAD+swizzle | 3007.2 | 55.2% |
| v4 double buf + f4st | 2839.9 | 52.1% |
| v5 warp tiling (未修复) | 1768.8 | 32.5% |
| Tensor Core naive | 1649.9 | 30.3% |
| **cuBLAS** | **5447.1** | **100%** |

之前 cuBLAS 跑出 2974 GFLOPS 是冷启动未充分 warmup，实际 warmup 后为 **5447.1 GFLOPS**（3.154 ms）。手写 FP32 kernel 最好成绩 (v3: 3007) 只达到 cuBLAS 的 55%，差距仍然很大。cuBLAS 内部使用了更激进的寄存器 tiling、向量化加载、warp 级流水线等深度优化。
