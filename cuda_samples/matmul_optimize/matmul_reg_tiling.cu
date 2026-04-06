#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ============================================================
// 寄存器 Tiling (Register Tiling) 优化
// ============================================================
//
// 核心思想：每个线程不再只计算 C 的 1 个元素，而是计算 TM×TN 个元素。
// 从 shared memory 加载到寄存器后，一个值被复用多次，大幅减少访存。
//
// ---- 参数说明 ----
//   BM, BN: Block Tile 大小 — 一个 thread block 负责 C 的 BM×BN 子块
//   BK:     K 维度的 tile 大小 — 每次从 A/B 加载 BK 列/行
//   TM, TN: Thread Tile 大小 — 每个线程负责 C 的 TM×TN 个元素
//
// ---- 线程块配置 ----
//   blockDim.x = BN / TN = 128 / 8 = 16
//   blockDim.y = BM / TM = 128 / 8 = 16
//   每个 block 共 256 个线程
//
// ---- 相比 shared memory 基础版的优化点 ----
//
// 1. 内积 → 外积（最关键）
//    基础版：每个线程取 A 的一行 · B 的一列 做点积（内积），算 1 个 C 元素
//    本版本：每个 bk 步取 A 的一列片段(TM个) × B 的一行片段(TN个) 做外积，
//           一次更新 TM×TN=64 个 C 元素
//    读 TM+TN=16 个 shared mem 值，做 TM×TN=64 次乘加
//    访存比从 1:2 提升到 4:1（提升 8 倍）
//
// 2. 寄存器缓存（多一层存储层级）
//    数据流: global memory → shared memory → 寄存器 → 计算
//    CUDA 中 kernel 的局部变量（float regA[TM] 等）由编译器自动分配到寄存器，
//    无需显式关键字。寄存器延迟 ~0 周期，shared memory ~20-30 周期。
//    本 kernel 每线程使用寄存器: regC[8][8] + regA[8] + regB[8] = 80 个 float
//    （硬件上限 255 个寄存器/线程，80 个在预算范围内，不会 spill）
//
// 3. 更大的 Block Tile（128×128 vs 32×32）
//    一个 block 覆盖 16384 个 C 元素（基础版仅 1024 个）
//    从 global memory 搬入的 A/B 数据被更多线程复用
//
// 4. 加载与计算的线程映射解耦
//    加载: 用 tid（一维编号）分配 → 优化 global memory coalescing
//    计算: 用 tx, ty（二维编号）分配 → 优化寄存器数据复用
//    两者独立设计，各自最优
//
// ---- 协作加载原理 ----
//   256 个线程按 行优先 映射到 shared memory 矩阵:
//     tid / 列数 = row,  tid % 列数 = col
//   让同一 warp 内连续线程访问相邻列 → global memory coalescing
//   As[128][8]:  一轮覆盖 256/8=32 行，需 4 轮 (stride=32)
//   Bs[8][128]:  一轮覆盖 256/128=2 行，需 4 轮 (stride=2)
//
// ---- CUDA 存储层级参考 ----
//   __shared__ float x   → shared memory  (~20-30 cycles)
//   float x (局部变量)    → 寄存器         (~0 cycles, 编译器自动分配)
//   局部数组过大溢出时     → local memory   (~几百 cycles, 实际在显存)
//
// ============================================================

#define BM 128    // Block tile M
#define BN 128    // Block tile N
#define BK 8      // Block tile K
#define TM 8      // Thread tile M
#define TN 8      // Thread tile N

__global__ void MatMulRegTilingKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K)
{
    // 每个 block 负责 C 的 [BM × BN] 子块
    // block 内线程排布：blockDim.x = BN/TN, blockDim.y = BM/TM
    const int bx = blockIdx.x;  // block 列号
    const int by = blockIdx.y;  // block 行号
    const int tx = threadIdx.x; // 线程列号 (0 ~ BN/TN-1)
    const int ty = threadIdx.y; // 线程行号 (0 ~ BM/TM-1)

    // 线程在 block 内的一维编号，用于协作加载数据
    const int tid = ty * blockDim.x + tx;
    const int numThreads = blockDim.x * blockDim.y; // = 256

    // Shared memory: 存放当前 tile 的 A 和 B 子块
    __shared__ float As[BM][BK];  // BM × BK = 128 × 8
    __shared__ float Bs[BK][BN];  // BK × BN = 8 × 128

    // 寄存器：每个线程的 TM×TN 个 C 累加值
    float regC[TM][TN] = {0.0f};

    // 寄存器：缓存从 shared memory 读取的 A 和 B 值
    float regA[TM];
    float regB[TN];

    // C 子块的全局起始位置
    const int cRow = by * BM;
    const int cCol = bx * BN;

    // A 子块的全局起始行，B 子块的全局起始列
    const float* aPtr = A + cRow * K;  // A[cRow, 0]
    const float* bPtr = B + cCol;      // B[0, cCol]

    // ========== 协作加载的索引预计算 ==========
    // 加载 As[BM][BK]: 共 BM*BK = 1024 个元素，256 个线程，每个线程加载 4 个
    const int loadARow = tid / BK;          // 该线程负责加载 As 的哪一行
    const int loadACol = tid % BK;          // 该线程负责加载 As 的哪一列
    const int strideA  = numThreads / BK;   // = 256/8 = 32，每轮加载覆盖 32 行

    // 加载 Bs[BK][BN]: 共 BK*BN = 1024 个元素，256 个线程，每个线程加载 4 个
    const int loadBRow = tid / BN;          // 该线程负责加载 Bs 的哪一行
    const int loadBCol = tid % BN;          // 该线程负责加载 Bs 的哪一列
    const int strideB  = numThreads / BN;   // = 256/128 = 2，每轮加载覆盖 2 行

    // ========== 沿 K 维度循环，每次处理 BK 列 ==========
    for (int k = 0; k < K; k += BK) {

        // ------ 协作加载 A tile 到 shared memory ------
        // As[BM][BK] = A[cRow..cRow+BM-1, k..k+BK-1]
        // 每个线程加载多行（strideA 步进）
        for (int i = 0; i < BM; i += strideA) {
            As[loadARow + i][loadACol] = aPtr[(loadARow + i) * K + k + loadACol];
        }

        // ------ 协作加载 B tile 到 shared memory ------
        // Bs[BK][BN] = B[k..k+BK-1, cCol..cCol+BN-1]
        for (int i = 0; i < BK; i += strideB) {
            Bs[loadBRow + i][loadBCol] = bPtr[(k + loadBRow + i) * N + loadBCol];
        }

        __syncthreads();

        // ------ 计算：从 shared memory 读取到寄存器，做外积累加 ------
        for (int bk = 0; bk < BK; ++bk) {
            // 加载 A 的一列到寄存器：As[ty*TM .. ty*TM+TM-1][bk]
            for (int tm = 0; tm < TM; ++tm) {
                regA[tm] = As[ty * TM + tm][bk];
            }
            // 加载 B 的一行到寄存器：Bs[bk][tx*TN .. tx*TN+TN-1]
            for (int tn = 0; tn < TN; ++tn) {
                regB[tn] = Bs[bk][tx * TN + tn];
            }
            // 外积：TM × TN 次乘加
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    regC[tm][tn] += regA[tm] * regB[tn];
                }
            }
        }

        __syncthreads();
    }

    // ========== 写回 C ==========
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            int globalRow = cRow + ty * TM + tm;
            int globalCol = cCol + tx * TN + tn;
            C[globalRow * N + globalCol] = regC[tm][tn];
        }
    }
}

// Host code
void MatMul(const float* A, const float* B, float* C,
            int M, int N, int K,
            float *transfer_ms, float *kernel_ms)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // H2D
    cudaEventRecord(e0);
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    // Kernel
    dim3 dimBlock(BN / TN, BM / TM);  // (16, 16) = 256 threads
    dim3 dimGrid(N / BN, M / BM);     // grid 覆盖整个 C 矩阵
    cudaEventRecord(e2);
    MatMulRegTilingKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(e3);

    // D2H
    cudaEventRecord(e4);
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, e0, e1);
    cudaEventElapsedTime(&kern_ms, e2, e3);
    cudaEventElapsedTime(&d2h_ms, e4, e5);
    *transfer_ms = h2d_ms + d2h_ms;
    *kernel_ms   = kern_ms;

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// CPU reference
void MatMulCPU(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main()
{
    // M, N, K 必须是 BM, BN, BK 的整数倍（128, 128, 8）
    int M = 1024, N = 1024, K = 1024;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float* A     = (float*)malloc(bytesA);
    float* B     = (float*)malloc(bytesB);
    float* C_gpu = (float*)malloc(bytesC);
    float* C_cpu = (float*)malloc(bytesC);

    for (int i = 0; i < M * K; ++i) A[i] = (float)(i % 10) * 0.1f;
    for (int i = 0; i < K * N; ++i) B[i] = (float)(i % 7)  * 0.1f;

    // GPU
    float transfer_ms = 0, kernel_ms = 0;
    MatMul(A, B, C_gpu, M, N, K, &transfer_ms, &kernel_ms);

    // CPU
    clock_t cpu_start = clock();
    MatMulCPU(A, B, C_cpu, M, N, K);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }

    // GFLOPS = 2*M*N*K / time_sec / 1e9
    double gflops = 2.0 * M * N * K / (kernel_ms * 1e-3) / 1e9;

    printf("=== Register Tiling GEMM ===\n");
    printf("Matrix size: %dx%dx%d\n", M, N, K);
    printf("Block tile: %dx%d, Thread tile: %dx%d, K tile: %d\n", BM, BN, TM, TN, BK);
    printf("Block threads: %dx%d = %d\n", BN/TN, BM/TM, (BN/TN)*(BM/TM));
    printf("GPU kernel time:   %.3f ms  (%.1f GFLOPS)\n", kernel_ms, gflops);
    printf("GPU transfer time: %.3f ms (H2D + D2H)\n", transfer_ms);
    printf("GPU total time:    %.3f ms\n", kernel_ms + transfer_ms);
    printf("CPU time:          %.3f ms\n", cpu_ms);
    printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms);
    printf("Max error (GPU vs CPU): %e\n", max_err);
    if (max_err < 1e-2f)
        printf("PASSED\n");
    else
        printf("FAILED (max_err = %e)\n", max_err);

    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    return 0;
}
