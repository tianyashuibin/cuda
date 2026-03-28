# CUDA 编程学习指南

## 目录

1. [前置知识](#1-前置知识)
2. [GPU 架构基础](#2-gpu-架构基础)
3. [CUDA 编程模型](#3-cuda-编程模型)
4. [性能优化](#4-性能优化)
5. [同步与通信](#5-同步与通信)
6. [调试与性能分析](#6-调试与性能分析)
7. [常用库](#7-常用库)
8. [进阶主题](#8-进阶主题)
9. [推荐学习顺序](#9-推荐学习顺序)
10. [参考资源](#10-参考资源)

---

## 1. 前置知识

在学习 CUDA 之前，需要掌握以下基础：

- **C/C++ 基础** — 指针、内存管理、函数、结构体、编译流程
- **并行计算概念** — 并发 vs 并行、线程、锁、同步原语
- **计算机体系结构** — CPU vs GPU 架构差异、内存层次结构、缓存原理

---

## 2. GPU 架构基础

### 硬件结构

- **SM（Streaming Multiprocessor）** — GPU 的基本计算单元，每个 SM 包含多个 CUDA Core
- **CUDA Core** — 执行浮点/整数运算的基本单元
- **Warp** — 32 个线程组成一个 Warp，是 SM 调度的最小单位

### 线程组织层次

```
Grid
 └── Block（由 blockDim 决定大小）
      └── Warp（32 个线程）
           └── Thread（最小执行单元）
```

### 内存层次（从快到慢）

| 内存类型 | 位置 | 生命周期 | 大小 |
|---------|------|---------|------|
| 寄存器 | 片上 | 线程 | ~256KB/SM |
| 共享内存 | 片上 | Block | 48~96KB/SM |
| L1/L2 缓存 | 片上 | 自动管理 | MB 级 |
| 全局内存 | 片外 DRAM | 应用 | GB 级 |
| 常量内存 | 片外（有缓存）| 应用 | 64KB |
| 纹理内存 | 片外（有缓存）| 应用 | GB 级 |

---

## 3. CUDA 编程模型

### 函数修饰符

| 修饰符 | 运行在 | 调用自 |
|--------|--------|--------|
| `__global__` | Device | Host（或 Device） |
| `__device__` | Device | Device |
| `__host__` | Host | Host |

### Kernel 启动语法

```cpp
kernel_name<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

### 内置变量

```cpp
threadIdx.x / .y / .z   // 线程在 Block 内的索引
blockIdx.x  / .y / .z   // Block 在 Grid 内的索引
blockDim.x  / .y / .z   // Block 的维度
gridDim.x   / .y / .z   // Grid 的维度
```

### 典型线程索引计算

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 1D
```

### 内存管理

```cpp
// 设备内存分配与释放
cudaMalloc((void**)&d_ptr, size);
cudaFree(d_ptr);

// 数据传输
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);

// 统一内存（自动迁移，适合入门）
cudaMallocManaged((void**)&ptr, size);
```

### 最小示例：向量加法

```cpp
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 启动：每个 block 256 线程
int blocks = (n + 255) / 256;
vectorAdd<<<blocks, 256>>>(d_a, d_b, d_c, n);
```

---

## 4. 性能优化

### 内存访问优化

- **合并访问（Coalesced Access）** — 同一 Warp 内的线程访问连续的内存地址，可合并为一次事务
- **共享内存（Shared Memory）** — 用 `__shared__` 声明，带宽远高于全局内存，适合数据复用场景
- **Bank Conflict** — 共享内存分 32 个 bank，多线程访问同一 bank 会串行化，需错开访问
- **Pinned Memory** — 页锁定内存，Host↔Device 传输速度更快

```cpp
__shared__ float smem[256];   // 声明共享内存
```

### 计算优化

- **避免 Warp Divergence** — 同一 Warp 内分支不一致会导致串行执行，尽量让线程走相同路径
- **提高占用率（Occupancy）** — 每个 SM 上活跃 Warp 数 / 最大 Warp 数，通过调整 blockDim 和寄存器用量来提高
- **数学函数** — 使用 `__fmaf`（融合乘加）、`__expf`（单精度近似）等快速内联函数

### 异步并发

```cpp
// 创建 Stream 实现异步执行
cudaStream_t stream;
cudaStreamCreate(&stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

---

## 5. 同步与通信

### Block 内同步

```cpp
__syncthreads();        // 等待 Block 内所有线程到达此处
__syncwarp();           // 等待 Warp 内所有线程（更轻量）
```

### 原子操作

```cpp
atomicAdd(&counter, 1);       // 原子加
atomicCAS(&val, old, newVal); // 原子比较并交换
atomicMax(&maxVal, val);      // 原子取最大值
```

### CUDA Events（计时 & 同步）

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

---

## 6. 调试与性能分析

### 错误处理

```cpp
// 检查 CUDA API 调用
#define CUDA_CHECK(call) {                                      \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(1);                                                \
    }                                                           \
}

CUDA_CHECK(cudaMalloc(&ptr, size));
cudaDeviceSynchronize();                  // 等待所有 kernel 完成
cudaGetLastError();                       // 获取最近错误
```

### 性能分析工具

| 工具 | 用途 |
|------|------|
| **Nsight Systems** | 系统级时间线分析，查看 CPU/GPU 交互 |
| **Nsight Compute** | Kernel 级深度分析，给出优化建议 |
| **compute-sanitizer** | 内存越界、竞态条件检测（替代旧版 cuda-memcheck） |
| **nvprof**（旧）| 命令行快速 profiling |

---

## 7. 常用库

| 库 | 功能 |
|----|------|
| **cuBLAS** | 基本线性代数（GEMM、向量点积等） |
| **cuFFT** | 快速傅里叶变换 |
| **cuDNN** | 深度学习算子（卷积、BN、激活等） |
| **cuSPARSE** | 稀疏矩阵运算 |
| **Thrust** | GPU 版 STL（排序、规约、扫描等），开发效率高 |
| **CUB** | Block/Warp 级底层原语，性能更可控 |
| **NCCL** | 多 GPU / 多节点集合通信 |

---

## 8. 进阶主题

- **Tensor Core** — 专用矩阵乘加单元（FP16/BF16/INT8），通过 WMMA API 或 cuBLAS 使用
- **CUDA Graph** — 将一系列 kernel 和内存操作录制为 Graph，减少启动开销
- **PTX 汇编** — CUDA 的中间表示，可内联优化热点代码
- **多 GPU 编程** — `cudaSetDevice`、Peer-to-Peer 访问、NVLink
- **自定义 PyTorch/TensorFlow 算子** — 将 CUDA kernel 集成到深度学习框架
- **动态并行（Dynamic Parallelism）** — Kernel 内部启动子 Kernel

---

## 9. 推荐学习顺序

```
阶段 1：基础
  C/C++ 基础 → GPU 架构概念 → CUDA 线程模型 → 内存管理
  实践：vector_add、matrix_multiply

阶段 2：优化
  合并访问 → 共享内存 → Warp Divergence → Occupancy 分析
  实践：优化矩阵乘法、并行规约

阶段 3：工程
  CUDA Streams → 错误处理 → Nsight 性能分析
  实践：流水线 Host/Device 重叠

阶段 4：进阶
  Tensor Core → CUDA Graph → 多 GPU → 集成到框架
  实践：实现自定义深度学习算子
```

---

## 10. 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 官方文档，最权威
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 官方优化指南
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/) — 官方课程
- *Programming Massively Parallel Processors* (Kirk & Hwu) — 经典教材
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog) — 技术博客，涵盖最新特性