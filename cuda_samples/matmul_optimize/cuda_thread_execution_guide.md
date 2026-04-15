# CUDA 线程执行模型 —— 以 Shared Memory MatMul 为例

以 `matmul_sharemem.cu` 为例，梳理 BLOCK_SIZE 设置、GPU 硬件映射、线程执行的完整关系。

---

## 1. `__global__` 函数：每个线程都会执行

```c
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
```

`__global__` 标记的函数（kernel）不是被调用一次，而是被 **所有线程各执行一次**。

### 启动方式与坐标约定

```c
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);  // 每个 block 的线程数
dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);  // block 的数量

MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
```

当 N=1024, BLOCK_SIZE=32 时：
- dimGrid = (32, 32) → 1024 个 block
- dimBlock = (32, 32) → 每个 block 1024 个线程
- **总计 1024 × 1024 = 1,048,576 个线程**，每个都执行 `MatMulKernel`

### CUDA 坐标约定：dim3 是 (x, y) = (列, 行)

**`dim3` 的参数顺序是 (x, y)，而矩阵习惯是 (行, 列)，两者相反。**

```
dim3 dimGrid(B.width / dimBlock.x,   // x 维度 → 列方向 → 结果矩阵的宽 → 由 B.width 决定
             A.height / dimBlock.y);  // y 维度 → 行方向 → 结果矩阵的高 → 由 A.height 决定
```

对应关系：

```
              blockIdx.x →  (列方向)
            ┌──────┬──────┬──────┬──────┐
            │(0,0) │(1,0) │(2,0) │(3,0) │
blockIdx.y  │(0,1) │(1,1) │(2,1) │(3,1) │
    ↓       │(0,2) │(1,2) │(2,2) │(3,2) │
 (行方向)    └──────┴──────┴──────┴──────┘

x → 水平 → 列 → 宽度（width）
y → 垂直 → 行 → 高度（height）
```

| CUDA 变量 | 含义 | 对应矩阵维度 |
|---|---|---|
| `dim3(x, y)` | 先列后行 | 先 width 后 height |
| `blockIdx.x` / `threadIdx.x` | 列方向（水平） | col |
| `blockIdx.y` / `threadIdx.y` | 行方向（垂直） | row |

> 简单记忆：**dim3 是图形学约定 (x=水平, y=垂直)，和矩阵的 (行, 列) 正好反过来**。
> 所以 dimGrid 中 B.width 写在前面（对应 x/列），A.height 写在后面（对应 y/行）。

### 线程如何区分自己的身份

每个线程通过内置变量知道"我是谁、我该算哪个元素"：

```c
int blockRow = blockIdx.y;   // y → 行方向
int blockCol = blockIdx.x;   // x → 列方向
int row = threadIdx.y;       // y → block 内的行位置
int col = threadIdx.x;       // x → block 内的列位置
```

100 万个线程执行 **完全相同的代码**，但 `blockIdx` 和 `threadIdx` 不同，导致读取不同的数据、写入不同的位置。

### 类比 CPU 思维

```
// CPU 伪代码（串行）
for (blockRow = 0..31)
  for (blockCol = 0..31)
    for (row = 0..31)
      for (col = 0..31)
        执行 MatMulKernel(...)   // 100 万次循环

// GPU 实际做的
同时启动 100 万个线程，每个线程执行一次 MatMulKernel
```

---

## 2. BLOCK_SIZE 设置与硬件限制

### 硬件硬性约束

| 约束项 | 限制 | 说明 |
|---|---|---|
| 每个 block 最大线程数 | **1024** | 所有 CUDA GPU 通用，不可突破 |
| 每个 SM 的 shared memory | **48~228 KB**（取决于架构） | block 内所有线程共享 |
| 每个 SM 最大 warp 数 | 48~64（取决于架构） | 影响 occupancy |

### BLOCK_SIZE 实际可选范围

由于 block 线程数 = BLOCK_SIZE × BLOCK_SIZE（二维），可选值非常有限：

| BLOCK_SIZE | 线程数/block | shared memory 用量 | 能否运行 |
|---|---|---|---|
| 8 | 64 | 0.5 KB | OK，但线程太少 |
| 16 | 256 | 2 KB | OK，常用 |
| 32 | 1024 | 8 KB | OK，刚好到上限 |
| **64** | **4096** | **32 KB** | **不行，线程数超 1024 上限** |
| **1024** | **1,048,576** | **8 MB** | **不行，远超所有限制** |

> shared memory 计算：2 × BLOCK_SIZE² × 4 bytes（As 和 Bs 两个 float 数组）

---

## 3. Block、Warp 与 SM 的硬件映射

### 三层结构

```
Grid（整个计算任务）
├── Block (0,0)  ──→ 被调度到 SM 0
├── Block (0,1)  ──→ 被调度到 SM 1
├── Block (1,0)  ──→ 被调度到 SM 0（同一个 SM 可运行多个 block）
└── ...
```

- 一个 block **只在一个 SM 上** 执行，不会跨 SM
- 一个 SM 可以同时驻留 **多个 block**，由资源（线程数、shared memory、寄存器）决定上限
- GPU 硬件自动调度，程序员不控制 block 到 SM 的映射

### Warp：硬件实际执行单位

GPU 不会逐线程执行，而是以 **warp（32 个线程）** 为最小调度单位。

以 16×16 block 为例（256 线程 = 8 warps）：

```
Block 16×16 (256 threads)
├── warp 0: thread (0,0)~(0,15), (1,0)~(1,15)    ← 第 0-1 行
├── warp 1: thread (2,0)~(2,15), (3,0)~(3,15)    ← 第 2-3 行
├── warp 2: thread (4,0)~(4,15), (5,0)~(5,15)    ← 第 4-5 行
├── warp 3: thread (6,0)~(6,15), (7,0)~(7,15)    ← 第 6-7 行
├── warp 4: thread (8,0)~(8,15), (9,0)~(9,15)    ← 第 8-9 行
├── warp 5: thread (10,0)~(10,15), (11,0)~(11,15) ← 第 10-11 行
├── warp 6: thread (12,0)~(12,15), (13,0)~(13,15) ← 第 12-13 行
└── warp 7: thread (14,0)~(14,15), (15,0)~(15,15) ← 第 14-15 行
```

> 二维 block 按行展平为一维（row-major），然后每 32 个线程编为一个 warp。
> 同一个 warp 内的 32 个线程 **锁步执行同一条指令**（SIMT 模型）。

---

## 4. 为什么 BLOCK_SIZE=16 或 32 性能最好

### 核心原理：延迟隐藏

GPU 性能来自于 **当一个 warp 等待内存时，切换到另一个 warp 继续执行**。SM 上驻留的 warp 越多，隐藏延迟的能力越强。

### 各 BLOCK_SIZE 对比

| | 8×8 | 16×16 | 32×32 |
|---|---|---|---|
| 线程数/block | 64 | 256 | 1024 |
| warp 数/block | 2 | 8 | 32 |
| shared memory/block | 0.5 KB | 2 KB | 8 KB |
| SM 可驻留 block 数 | 多 | 中等 | 少（1~2 个） |
| SM 总 warp 数 | 少（block 内 warp 太少） | **多（平衡最优）** | 中等（block 太大，驻留数少） |

**关键权衡：**

- **BLOCK_SIZE 太小**（8×8）：每个 block 只有 2 个 warp，虽然 SM 能装很多 block，但总 warp 可能受限于其他资源（寄存器等），且 block 调度本身有开销
- **BLOCK_SIZE=16**：8 warps/block，SM 能驻留多个 block，总 warp 数充足，延迟隐藏效果好
- **BLOCK_SIZE=32**：32 warps/block 已占满线程上限，SM 可能只能驻留 1-2 个 block，灵活性差

> 实际最优值取决于具体 GPU 架构和 kernel 的寄存器/shared memory 使用量。
> 推荐用 NVIDIA 的 **Occupancy Calculator** 或 `--ptxas-options=-v` 编译选项来分析。

---

## 5. 完整执行流程图

以 N=1024, BLOCK_SIZE=16 为例：

```
Host（CPU）                          Device（GPU）
    |                                     |
    |  MatMulKernel<<<(64,64),(16,16)>>>  |
    |  ─────────────────────────────────→ |
    |                                     |
    |                          Grid: 64×64 = 4096 个 block
    |                          每个 block: 16×16 = 256 个线程
    |                          总计: 4096 × 256 = 1,048,576 个线程
    |                                     |
    |                          ┌─ SM 0: block(0,0), block(0,1), ...
    |                          ├─ SM 1: block(1,0), block(1,1), ...
    |                          ├─ ...
    |                          └─ SM N: ...
    |                                     |
    |                          每个 block 内部：
    |                          ┌─ warp 0 (thread 0~31)
    |                          ├─ warp 1 (thread 32~63)
    |                          ├─ ...
    |                          └─ warp 7 (thread 224~255)
    |                                     |
    |                          每个线程：
    |                          1. 用 blockIdx + threadIdx 确定自己负责的元素
    |                          2. 循环加载子矩阵到 shared memory
    |                          3. __syncthreads() 同步
    |                          4. 计算局部点积
    |                          5. 写回结果到全局内存
    |                                     |
    |  cudaMemcpy(D2H)                    |
    | ←───────────────────────────────────|
```

---

## 6. SM 内部结构与 Warp 调度

### SM 的 4 个分区（Sub-partition）

以 Ampere 架构（如 A100/RTX 3090）为例：

```
SM
├── Sub-partition 0:  1 个 Warp Scheduler + 32 FP32 cores + Register File
├── Sub-partition 1:  1 个 Warp Scheduler + 32 FP32 cores + Register File
├── Sub-partition 2:  1 个 Warp Scheduler + 32 FP32 cores + Register File
└── Sub-partition 3:  1 个 Warp Scheduler + 32 FP32 cores + Register File

共 4 个 Warp Scheduler，128 个 FP32 cores
最大驻留 64 warps → 每个 Scheduler 分配 16 个 warp
```

### 驻留 vs 执行

16 个 warp 分配给一个 Scheduler，但 **不是同时执行**：

```
Warp Scheduler 0（管理 16 个 warp）：

时钟周期 1:  warp 0  发射指令 → 32 个 FP32 cores 执行
时钟周期 2:  warp 3  发射指令（warp 0 在等内存）
时钟周期 3:  warp 7  发射指令（warp 3 在等内存）
时钟周期 4:  warp 0  发射指令（数据回来了）
...
```

- 每个 Scheduler 每个周期只能从 **1 个 warp 发射指令**
- 32 个 FP32 cores 刚好处理 1 个 warp（32 个线程）
- 其余 15 个 warp 处于 **就绪/等待状态**，随时可以切入
- 整个 SM 的 4 个 Scheduler **并行工作**，每周期最多发射 4 条指令，驱动全部 128 个 FP32 cores

---

## 7. BLOCK_SIZE 对调度灵活性的影响

### 层面一：SM 内部调度（主要问题）

问题出在 `__syncthreads()`——block 内所有线程必须到达同步点才能继续。

**BLOCK_SIZE=32（SM 上 2 个 block）：**

```
Block A 的 32 warps: 全部到达 syncthreads → 全部暂停
Block B 的 32 warps: 还在跑

→ 一瞬间 SM 上只有 1 个 block 的 warp 可调度
→ 如果 Block B 也到达 syncthreads，SM 完全空转！
```

**BLOCK_SIZE=16（SM 上 8 个 block）：**

```
Block A 的 8 warps: 到达 syncthreads → 暂停
Block B 的 8 warps: 还在跑
Block C 的 8 warps: 还在跑
Block D 的 8 warps: 还在跑
... 还有 4 个 block

→ 任何时候都有大量 warp 可调度，SM 几乎不会空转
```

### 层面二：跨 SM 分配（小矩阵时才是问题）

以 N=1024、80 个 SM 的 GPU 为例：

| BLOCK_SIZE | Grid 大小 | Block 总数 | 每个 SM 分到 |
|---|---|---|---|
| 16 | 64×64 | 4096 | ~51 个 block |
| 32 | 32×32 | 1024 | ~12 个 block |

两种情况 block 数都远多于 SM 数，所有 SM 都能被喂饱。

但如果矩阵很小（N=128）：

| BLOCK_SIZE | Block 总数 | 80 个 SM 的情况 |
|---|---|---|
| 16 | 64 | 还不够每个 SM 分一个 |
| 32 | 16 | 只有 16 个 SM 在工作，其余 64 个空闲 |

这叫 **wave quantization（波次量化）** 问题——block 太少导致部分 SM 闲置。

### 总结

| | 跨 SM 分配 | SM 内部调度 |
|---|---|---|
| 问题场景 | 矩阵小导致 block 总数少于 SM 数 | `__syncthreads()` 导致整个 block 的 warp 同时暂停 |
| BLOCK_SIZE=32 | block 少，小矩阵时部分 SM 空闲 | SM 只有 2 个 block，同步时可调度 warp 骤减 |
| BLOCK_SIZE=16 | block 多，分配更均匀 | SM 有 8 个 block，同步时仍有大量 warp 可用 |

---

## 8. Occupancy（占用率）

### 定义

**SM 上实际驻留的 warp 数 / SM 最大能驻留的 warp 数。**

### 为什么重要：延迟隐藏

GPU 隐藏延迟的唯一手段就是 **切换 warp**。当 warp A 在等内存数据时，硬件立刻切到 warp B 执行，零开销切换。

```
时间线（一个 SM 上）：

warp 0: [计算][等内存...............][计算]
warp 1:       [计算][等内存...............][计算]
warp 2:             [计算][等内存...............][计算]
warp 3:                   [计算][计算][等内存...........]

→ SM 始终在忙，没有空闲周期
```

如果 SM 上只有 1-2 个 warp，内存延迟就无法被掩盖，SM 就会空转等待。

### 限制 occupancy 的三大资源

SM 上能驻留多少 warp，取决于三个资源 **谁先用完**：

| 资源 | SM 上限（典型值） | 消耗方 |
|---|---|---|
| 线程数 | 2048（= 64 warps） | 每个 block 的线程数 |
| Shared memory | 48~228 KB | kernel 中 `__shared__` 声明的数组 |
| 寄存器 | 65536 个 | kernel 中局部变量越多，每线程用的寄存器越多 |

**三者中任何一个先耗尽，就不能再塞更多 block 了。**

### 以 matmul_sharemem 为例

假设 SM 最大 2048 线程（64 warps），shared memory 48 KB：

**BLOCK_SIZE=16（256 线程/block，shared memory 2 KB/block）：**
- 线程维度：2048 / 256 = 可放 **8 个 block**
- shared memory：48 KB / 2 KB = 可放 **24 个 block**
- 瓶颈是线程数 → 驻留 8 个 block = 64 warps → **occupancy = 64/64 = 100%**

**BLOCK_SIZE=32（1024 线程/block，shared memory 8 KB/block）：**
- 线程维度：2048 / 1024 = 可放 **2 个 block**
- shared memory：48 KB / 8 KB = 可放 **6 个 block**
- 瓶颈是线程数 → 驻留 2 个 block = 64 warps → **occupancy = 64/64 = 100%**

两者 occupancy 相同，但 BLOCK_SIZE=16 的 8 个 block 比 2 个大 block **调度更灵活**——如果某个 block 中所有 warp 都在等 `__syncthreads()`，其他 block 的 warp 还能继续跑。

### occupancy 不是越高越好

100% occupancy 不一定最快。有时候降低 occupancy 换取：
- 每线程更多寄存器 → 减少寄存器溢出到慢速内存
- 更多 shared memory → 减少全局内存访问

**实际调优需要 profiling**，NVIDIA 提供了工具：

```bash
# 编译时查看寄存器/shared memory 使用量
nvcc --ptxas-options=-v matmul_sharemem.cu

# 用 nsight compute 分析实际 occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./a.out
```

---

## 9. 关键概念总结

| 概念 | 说明 |
|---|---|
| `__global__` | 标记 kernel 函数，每个线程各执行一次 |
| `threadIdx` | 线程在 block 内的坐标，区分"我算哪个元素" |
| `blockIdx` | block 在 grid 内的坐标，区分"我负责哪个子矩阵" |
| Block | 一组线程，共享 shared memory，在同一个 SM 上执行 |
| Warp（32 线程） | 硬件实际调度的最小单位，锁步执行 |
| SM | 流多处理器，执行 block 的硬件单元 |
| Occupancy | SM 上实际驻留 warp 数 / 最大可驻留 warp 数，越高越能隐藏延迟 |
| BLOCK_SIZE | 决定线程数、shared memory 用量、occupancy 的关键参数 |
