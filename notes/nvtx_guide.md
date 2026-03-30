# NVTX 使用指南

NVTX（NVIDIA Tools Extension Library）是 NVIDIA 提供的一套轻量级 C/C++ API，允许在代码中插入自定义标注，这些标注会出现在 Nsight Systems 的时间线上，让你能将性能数据与具体的代码逻辑对应起来。

---

## 为什么需要 NVTX

默认的 profiling 时间线只显示 CUDA API 调用和 kernel 名称，当程序有多个阶段（数据加载、预处理、推理、后处理）时，难以区分各阶段耗时。NVTX 让你能：

- 给代码段加上有颜色的命名区间，时间线上一目了然
- 标记关键时间点（如某次迭代开始）
- 给 CUDA stream 命名，便于在时间线上区分不同流

---

## 两套 API 对比

| API | 头文件 | 链接 | 特点 |
|-----|--------|------|------|
| **NVTX3 C++ API**（推荐） | `nvtx3/nvtx3.hpp` | 无需额外链接 | Header-only，CUDA 10.0+ 内置，支持 RAII |
| NVTX C API（旧版） | `nvToolsExt.h` | `-lnvToolsExt` | 需要手动 push/pop，更底层 |

CUDA Toolkit 已内置 NVTX3，优先使用 NVTX3 C++ API。

---

## NVTX3 C++ API（推荐）

### 编译

```bash
# 只需包含头文件，无需额外链接
nvcc -O2 -lineinfo -o build/my_program my_program.cu
```

### 基本用法：scoped_range

`scoped_range` 是 RAII 风格，离开作用域自动结束区间，最常用。

```cpp
#include <nvtx3/nvtx3.hpp>

void train_step() {
    nvtx3::scoped_range range{"train_step"};  // 区间开始
    // ... 代码 ...
}   // 离开作用域，区间自动结束
```

### 带颜色的区间

```cpp
#include <nvtx3/nvtx3.hpp>

// 预定义颜色
nvtx3::scoped_range r1{"data_load",   nvtx3::rgb{0,   255, 0}};   // 绿色
nvtx3::scoped_range r2{"forward",     nvtx3::rgb{0,   128, 255}}; // 蓝色
nvtx3::scoped_range r3{"backward",    nvtx3::rgb{255, 128, 0}};   // 橙色
nvtx3::scoped_range r4{"optimizer",   nvtx3::rgb{255, 0,   0}};   // 红色
```

### 标记单个时间点

```cpp
nvtx3::mark("checkpoint_reached");
```

### 给 CUDA stream 命名

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
nvtx3::set_name(stream, "compute_stream");
```

---

## NVTX C API（旧版）

### 编译

```bash
nvcc -O2 -lineinfo -o build/my_program my_program.cu -lnvToolsExt
```

### Push/Pop 区间（支持嵌套）

```cpp
#include <nvToolsExt.h>

nvtxRangePushA("data_transfer");   // 开始区间
cudaMemcpy(...);
nvtxRangePop();                    // 结束区间
```

### 带颜色的区间

```cpp
#include <nvToolsExt.h>

// 定义带颜色的属性
nvtxEventAttributes_t attr = {};
attr.version       = NVTX_VERSION;
attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attr.colorType     = NVTX_COLOR_ARGB;
attr.color         = 0xFF00FF00;          // ARGB：不透明绿色
attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
attr.message.ascii = "forward_pass";

nvtxRangePushEx(&attr);
// ... kernel 调用 ...
nvtxRangePop();
```

### 非嵌套区间（Start/End）

```cpp
nvtxRangeId_t id = nvtxRangeStartA("async_region");
// ... 可以在任意位置结束，不需要对称嵌套 ...
nvtxRangeEnd(id);
```

### 标记时间点

```cpp
nvtxMarkA("iteration_start");
```

### 给 CUDA stream / thread 命名

```cpp
// 命名 CUDA stream（C API）
nvtxNameCudaStreamA(stream, "data_stream");

// 命名当前线程
nvtxNameOsThreadA(pthread_self(), "worker_thread");
```

---

## 完整示例

```cpp
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdio.h>

__global__ void kernel_a(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

__global__ void kernel_b(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

int main() {
    const int N = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // 整体训练循环
    for (int iter = 0; iter < 3; iter++) {
        nvtx3::scoped_range loop_range{
            ("iteration_" + std::to_string(iter)).c_str()
        };

        // 阶段 1：数据准备（绿色）
        {
            nvtx3::scoped_range r{"data_prep", nvtx3::rgb{0, 200, 0}};
            float* h_data = new float[N];
            cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_data;
        }

        // 阶段 2：kernel A（蓝色）
        {
            nvtx3::scoped_range r{"kernel_a", nvtx3::rgb{0, 128, 255}};
            kernel_a<<<(N+255)/256, 256>>>(d_data, N);
        }

        // 阶段 3：kernel B（橙色）
        {
            nvtx3::scoped_range r{"kernel_b", nvtx3::rgb{255, 128, 0}};
            kernel_b<<<(N+255)/256, 256>>>(d_data, N);
        }

        nvtx3::mark("iteration_done");
    }

    cudaFree(d_data);
    return 0;
}
```

---

## 在 Nsight Systems 中查看 NVTX

```bash
# 采集时需要加 --trace=nvtx（或 cuda,nvtx）
nsys profile --trace=cuda,nvtx -o my_report ./build/my_program

# 命令行查看 NVTX 区间统计
nsys stats my_report.nsys-rep
```

在 Nsight Systems GUI 时间线中，NVTX 区间会显示为：
- 带颜色的色块，悬浮显示名称和持续时间
- 可与下方的 CUDA kernel / memcpy 行对齐，直观看出各阶段对应的 GPU 活动

---

## 常用颜色参考（NVTX3 rgb）

| 颜色 | 代码 | 建议用途 |
|------|------|---------|
| 绿色 | `{0, 200, 0}` | 数据加载 / IO |
| 蓝色 | `{0, 128, 255}` | 前向计算 |
| 橙色 | `{255, 128, 0}` | 反向传播 |
| 红色 | `{255, 0, 0}` | 梯度更新 / 关键路径 |
| 紫色 | `{160, 0, 255}` | 通信 / AllReduce |
| 黄色 | `{255, 220, 0}` | 预处理 |

---

## 本项目使用建议

| 文件 | 建议加 NVTX 的位置 |
|------|-------------------|
| `cuda_stream_demo.cu` | 标记每个 stream 的 H2D / kernel / D2H 三个阶段 |
| `cuda_graph_demo.cu` | 标记 graph capture 阶段和 graph replay 阶段 |
| `matmul_*.cu` | 标记数据初始化、kernel 执行、结果验证三段 |

> **注意**：NVTX 本身开销极低（纳秒级），不会影响 profiling 结果的准确性。Release build 中保留 NVTX 标注不影响性能。
