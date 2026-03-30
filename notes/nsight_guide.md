# Nsight 使用指南

Nsight 是 NVIDIA 官方的 GPU 性能分析工具套件，包含两个主要工具：

| 工具 | 用途 |
|------|------|
| **Nsight Systems** (`nsys`) | 系统级时间线分析，查看 CPU/GPU 整体交互 |
| **Nsight Compute** (`ncu`) | Kernel 级深度分析，给出具体性能瓶颈和优化建议 |

两者配合使用：先用 Nsight Systems 找到耗时的 kernel，再用 Nsight Compute 深入分析该 kernel。

---

## 编译要求

Nsight 分析需要在编译时加 `-lineinfo`，让工具能将性能数据对应到源码行号：

```bash
nvcc -O2 -lineinfo -o build/matmul_sharemem matmul_sharemem.cu
# 或直接 make（本项目 Makefile 已包含 -lineinfo）
make
```

> **注意**：不要用 `-G`（设备端调试符号），它会禁用 GPU 优化，profiling 结果失真。

---

## Nsight Systems

### 基本用法

```bash
# 生成 .nsys-rep 报告文件
nsys profile ./build/cuda_stream_demo

# 指定输出文件名
nsys profile -o stream_report ./build/cuda_stream_demo

# 同时采集 CUDA API 调用 + kernel + 内存传输
nsys profile --trace=cuda,nvtx ./build/cuda_graph_demo
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `-o <name>` | 指定输出文件名（默认 report1） |
| `--trace=cuda` | 采集 CUDA API 和 kernel 事件 |
| `--trace=cuda,nvtx` | 额外采集 NVTX 自定义标记 |
| `--stats=true` | 命令行直接打印统计摘要 |

### 查看结果

```bash
# 命令行打印摘要（不需要 GUI）
nsys stats report1.nsys-rep

# 下载 .nsys-rep 文件到本地，用 Nsight Systems GUI 打开
# GUI 下载地址：https://developer.nvidia.com/nsight-systems
```

### 适合分析的问题

- kernel 和 memcpy 是否有重叠（stream 是否生效）
- CPU launch overhead 占比多少
- 各 kernel 的执行时间占比

---

## Nsight Compute

### 基本用法

```bash
# 采集完整性能指标（数据较多，速度慢）
ncu --set full ./build/matmul_sharemem

# 快速采集（默认指标集，速度快）
ncu ./build/matmul_sharemem

# 指定输出文件（生成 .ncu-rep，可用 GUI 打开）
ncu --set full -o matmul_report ./build/matmul_sharemem
```

### 只分析特定 kernel

```bash
# 按 kernel 函数名过滤
ncu --kernel-name MatMulKernel ./build/matmul_sharemem

# 只分析第 N 次 kernel 调用（从 1 开始）
ncu --launch-skip 0 --launch-count 1 ./build/matmul_sharemem
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--set full` | 采集全套指标（含 roofline 数据）|
| `--set detailed` | 详细指标，比 full 少一些 |
| `--kernel-name <name>` | 只分析指定名称的 kernel |
| `--launch-skip <n>` | 跳过前 n 次 kernel 调用 |
| `--launch-count <n>` | 只分析 n 次 kernel 调用 |
| `-o <name>` | 输出 .ncu-rep 文件 |

### 命令行查看关键指标

```bash
# 只看内存相关指标
ncu --metrics l1tex__t_bytes,lts__t_bytes,dram__bytes \
    ./build/matmul_sharemem

# 只看计算利用率
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/matmul_sharemem
```

### 常用分析指标说明

| 指标 | 含义 | 理想值 |
|------|------|--------|
| SM Active Cycles | SM 实际工作时间占比 | 越高越好 |
| Memory Throughput | 显存带宽利用率 | 接近峰值 |
| Compute Throughput | 计算单元利用率 | 接近峰值 |
| Achieved Occupancy | 实际 warp 占用率 | 越高越好 |
| L1/L2 Hit Rate | 缓存命中率 | 越高越好 |

### 适合分析的问题

- kernel 是 memory-bound 还是 compute-bound（Roofline 模型）
- shared memory 使用是否有 bank conflict
- global memory 访问是否合并（coalesced）
- occupancy 是否受寄存器或 shared memory 限制

---

## 在 Colab 上的完整流程

```bash
# 1. 编译（已含 -lineinfo）
cd /content/drive/MyDrive/cuda/samples/matmul_optimize
make

# 2. 用 Nsight Systems 看整体时间线
nsys profile -o report ./build/matmul_sharemem
nsys stats report.nsys-rep   # 命令行查看摘要

# 3. 用 Nsight Compute 深入分析某个 kernel
ncu --set full -o matmul_ncu ./build/matmul_sharemem

# 4. 下载 .ncu-rep / .nsys-rep 到本地，用 GUI 打开查看详细报告
```

---

## 本项目各文件适合用哪个工具

| 文件 | 推荐工具 | 关注点 |
|------|---------|--------|
| `matmul_sharemem` | Nsight Compute | shared memory 效率、occupancy |
| `matmul_cublas` | Nsight Compute | 与手写 kernel 对比吞吐量 |
| `matmul_tensorcore` | Nsight Compute | Tensor Core 利用率 |
| `cuda_stream_demo` | Nsight Systems | H2D/kernel/D2H 是否真正重叠 |
| `cuda_graph_demo` | Nsight Systems | 对比 launch overhead 减少情况 |
