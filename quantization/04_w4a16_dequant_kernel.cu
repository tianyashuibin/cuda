/*
 * 04_w4a16_dequant_kernel.cu — CUDA INT4 Dequantize + GEMM Kernel
 * ================================================================
 *
 * 这是量化推理中最核心的 CUDA kernel:
 *   - 权重以 INT4 packed 格式存储 (2 个 INT4 打包在 1 个 uint8 中)
 *   - 激活保持 FP16
 *   - kernel 内部做 dequantize (INT4 → FP16) 并与 matmul 融合
 *
 * 本文件包含:
 *   1. INT4 打包/解包工具函数
 *   2. 简单的 W4A16 dequantize-only kernel
 *   3. 简化版 W4A16 GEMM kernel (dequant + matmul fused)
 *
 * 编译: nvcc -O2 -arch=sm_80 04_w4a16_dequant_kernel.cu -o w4a16_demo
 * 运行: ./w4a16_demo
 *
 * 注意: 这是教学用的简化版本，生产级实现见 Marlin kernel。
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ================================================================
// 宏定义和工具函数
// ================================================================

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


// ================================================================
// 第一部分: INT4 打包和解包
// ================================================================

/*
 * INT4 数据格式:
 *
 * 一个 uint8 (8 bit) 存储 2 个 INT4 值:
 *   packed = (val_low & 0xF) | (val_high << 4)
 *
 * 解包:
 *   val_low  = packed & 0xF          (取低 4 bit)
 *   val_high = (packed >> 4) & 0xF   (取高 4 bit)
 *
 * 有符号 INT4 范围: -8 ~ 7
 *   如果无符号值 >= 8，则真实值 = 无符号值 - 16
 *
 * 一个 uint32 (32 bit) 可以存 8 个 INT4 值:
 *   val[i] = (packed >> (i * 4)) & 0xF
 */


// CPU 端: 将 float 权重量化为 packed INT4
void quantize_and_pack_int4(
    const float* weight,    // [K, N] float 权重
    uint8_t* packed,        // [K, N/2] packed INT4 输出
    float* scale,           // [K, N/group_size] scale
    int K, int N,
    int group_size
) {
    int num_groups_per_row = N / group_size;

    for (int k = 0; k < K; k++) {
        for (int g = 0; g < num_groups_per_row; g++) {
            // 计算这个 group 的 scale
            float abs_max = 0.0f;
            for (int j = 0; j < group_size; j++) {
                int n = g * group_size + j;
                float val = fabsf(weight[k * N + n]);
                if (val > abs_max) abs_max = val;
            }
            float s = abs_max / 7.0f;  // INT4 signed: max = 7
            if (s < 1e-8f) s = 1e-8f;
            scale[k * num_groups_per_row + g] = s;

            // 量化并打包
            for (int j = 0; j < group_size; j += 2) {
                int n = g * group_size + j;

                // 量化两个值
                int q0 = (int)roundf(weight[k * N + n] / s);
                int q1 = (int)roundf(weight[k * N + n + 1] / s);

                // Clamp 到 [-8, 7]
                q0 = max(-8, min(7, q0));
                q1 = max(-8, min(7, q1));

                // 转为无符号 4-bit 表示
                uint8_t u0 = (uint8_t)(q0 & 0xF);
                uint8_t u1 = (uint8_t)(q1 & 0xF);

                // 打包: 低位在低 4 bit，高位在高 4 bit
                packed[k * (N / 2) + n / 2] = u0 | (u1 << 4);
            }
        }
    }
}


// CPU 端: 解包并反量化 (用于验证)
void unpack_and_dequantize_int4(
    const uint8_t* packed,  // [K, N/2]
    const float* scale,     // [K, N/group_size]
    float* output,          // [K, N]
    int K, int N,
    int group_size
) {
    int num_groups_per_row = N / group_size;

    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n += 2) {
            uint8_t p = packed[k * (N / 2) + n / 2];

            // 解包
            int q0 = p & 0xF;
            int q1 = (p >> 4) & 0xF;

            // 符号扩展: 如果 >= 8，则减 16
            if (q0 >= 8) q0 -= 16;
            if (q1 >= 8) q1 -= 16;

            // 获取 scale
            int g = n / group_size;
            float s = scale[k * num_groups_per_row + g];

            // 反量化
            output[k * N + n] = (float)q0 * s;
            output[k * N + n + 1] = (float)q1 * s;
        }
    }
}


// ================================================================
// 第二部分: CUDA Dequantize Kernel (INT4 → FP16)
// ================================================================

/*
 * 将 packed INT4 权重解包并转换为 FP16
 * 这个 kernel 单独做 dequantize，不包含 matmul
 * 用于理解 INT4 解包在 GPU 上如何实现
 */
__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ packed,  // [K, N/2] packed INT4
    const float* __restrict__ scale,     // [K, N/group_size]
    half* __restrict__ output,           // [K, N] FP16 输出
    int K, int N,
    int group_size
) {
    // 每个线程处理 2 个 INT4 值 (即 1 个 uint8)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_packed = K * (N / 2);

    if (idx >= total_packed) return;

    // 确定位置
    int k = idx / (N / 2);          // 哪一行
    int packed_col = idx % (N / 2); // packed 列索引
    int n = packed_col * 2;         // 实际列索引

    // 读取 packed value
    uint8_t p = packed[idx];

    // 解包
    int q0 = (int)(p & 0xF);
    int q1 = (int)((p >> 4) & 0xF);

    // 符号扩展
    if (q0 >= 8) q0 -= 16;
    if (q1 >= 8) q1 -= 16;

    // 获取 scale
    int num_groups_per_row = N / group_size;
    int g = n / group_size;
    float s = scale[k * num_groups_per_row + g];

    // 反量化并写出 FP16
    output[k * N + n]     = __float2half((float)q0 * s);
    output[k * N + n + 1] = __float2half((float)q1 * s);
}


// ================================================================
// 第三部分: 简化版 W4A16 GEMM Kernel
// ================================================================

/*
 * C[M, N] = A[M, K] @ dequant(B_packed[K, N/2], scale[K, N/gs])
 *
 * 这是一个教学用的简化版本:
 *   - 每个线程计算输出矩阵的一个元素
 *   - 在 K 维度循环中: 加载 INT4 权重 → 解包 → dequant → FP16 乘加
 *   - 没有用 shared memory 或 Tensor Core (生产版本会用)
 *
 * 性能不是目标，理解数据流是目标。
 */
__global__ void w4a16_gemm_naive(
    const half* __restrict__ A,          // [M, K] FP16 激活
    const uint8_t* __restrict__ B_packed, // [K, N/2] packed INT4 权重
    const float* __restrict__ scale,      // [K, N/group_size] scale
    half* __restrict__ C,                 // [M, N] FP16 输出
    int M, int N, int K,
    int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M 维度
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N 维度

    if (row >= M || col >= N) return;

    int num_groups_per_row = N / group_size;
    float acc = 0.0f;  // FP32 累加

    for (int k = 0; k < K; k++) {
        // 加载 A (FP16 → FP32)
        float a_val = __half2float(A[row * K + k]);

        // 加载并解包 B (INT4 → FP32)
        int packed_idx = k * (N / 2) + col / 2;
        uint8_t p = B_packed[packed_idx];

        int q;
        if (col % 2 == 0) {
            q = (int)(p & 0xF);
        } else {
            q = (int)((p >> 4) & 0xF);
        }
        // 符号扩展
        if (q >= 8) q -= 16;

        // 获取 scale
        int g = col / group_size;
        float s = scale[k * num_groups_per_row + g];

        // Dequantize
        float b_val = (float)q * s;

        // 乘加
        acc += a_val * b_val;
    }

    C[row * N + col] = __float2half(acc);
}


/*
 * 优化版: 使用 shared memory 的 W4A16 GEMM
 *
 * 优化点:
 *   1. Tiling: 将 A 和 B 的 tile 加载到 shared memory
 *   2. 在 shared memory 中做 dequantize (减少 global memory 访问)
 *   3. 循环展开
 *
 * 仍然是简化版本，生产级实现还需要:
 *   - Tensor Core (WMMA/MMA)
 *   - Async copy (cp.async)
 *   - Software pipelining (double buffering)
 *   - Register tiling
 *
 * 这些是 Marlin kernel 的核心优化技巧。
 */
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void w4a16_gemm_tiled(
    const half* __restrict__ A,           // [M, K]
    const uint8_t* __restrict__ B_packed, // [K, N/2]
    const float* __restrict__ scale,      // [K, N/group_size]
    half* __restrict__ C,                 // [M, N]
    int M, int N, int K,
    int group_size
) {
    // Block 负责计算 C 的一个 [TILE_M, TILE_N] tile
    int bx = blockIdx.x;  // N 维度
    int by = blockIdx.y;  // M 维度
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory
    __shared__ float As[TILE_M][TILE_K];  // 激活 tile (FP32)
    __shared__ float Bs[TILE_K][TILE_N];  // 权重 tile (dequantized FP32)

    // 输出累加器
    float acc = 0.0f;

    int num_groups_per_row = N / group_size;

    // 沿 K 维度分 tile
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // ---- 加载 A tile 到 shared memory ----
        int a_row = by * TILE_M + ty;
        int a_col = t * TILE_K + tx;
        if (a_row < M && a_col < K) {
            As[ty][tx] = __half2float(A[a_row * K + a_col]);
        } else {
            As[ty][tx] = 0.0f;
        }

        // ---- 加载 B tile: 解包 INT4 → FP32 到 shared memory ----
        int b_row = t * TILE_K + ty;  // K 维度
        int b_col = bx * TILE_N + tx; // N 维度
        if (b_row < K && b_col < N) {
            // 从 packed INT4 解包
            int packed_idx = b_row * (N / 2) + b_col / 2;
            uint8_t p = B_packed[packed_idx];

            int q;
            if (b_col % 2 == 0) {
                q = (int)(p & 0xF);
            } else {
                q = (int)((p >> 4) & 0xF);
            }
            if (q >= 8) q -= 16;

            // Dequantize
            int g = b_col / group_size;
            float s = scale[b_row * num_groups_per_row + g];
            Bs[ty][tx] = (float)q * s;
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // ---- 计算 tile 乘法 ----
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写回
    int c_row = by * TILE_M + ty;
    int c_col = bx * TILE_N + tx;
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = __float2half(acc);
    }
}


// ================================================================
// 主函数: 测试和验证
// ================================================================

int main() {
    printf("================================================================\n");
    printf("04 W4A16 Dequantize CUDA Kernel Demo\n");
    printf("================================================================\n\n");

    // 参数
    int M = 32;          // batch * seq_len
    int K = 256;         // in_features
    int N = 256;         // out_features
    int group_size = 64; // per-group quantization

    printf("M=%d, K=%d, N=%d, group_size=%d\n", M, K, N, group_size);
    printf("INT4 权重大小: %.1f KB (vs FP16: %.1f KB, 压缩 4x)\n\n",
           K * N * 0.5f / 1024, K * N * 2.0f / 1024);

    // ---- 分配 Host 内存 ----
    int num_groups_per_row = N / group_size;

    float* h_weight = (float*)malloc(K * N * sizeof(float));
    float* h_A = (float*)malloc(M * K * sizeof(float));
    uint8_t* h_packed = (uint8_t*)malloc(K * (N / 2) * sizeof(uint8_t));
    float* h_scale = (float*)malloc(K * num_groups_per_row * sizeof(float));
    float* h_dequant = (float*)malloc(K * N * sizeof(float));

    // 初始化权重 (模拟真实分布)
    srand(42);
    for (int i = 0; i < K * N; i++) {
        h_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
    }
    for (int i = 0; i < M * K; i++) {
        h_A[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    // ---- Step 1: CPU 端量化 + 打包 ----
    printf("--- Step 1: CPU 端 INT4 量化和打包 ---\n");
    quantize_and_pack_int4(h_weight, h_packed, h_scale, K, N, group_size);

    // 验证: 解包并对比
    unpack_and_dequantize_int4(h_packed, h_scale, h_dequant, K, N, group_size);

    float max_quant_err = 0.0f;
    for (int i = 0; i < K * N; i++) {
        float err = fabsf(h_weight[i] - h_dequant[i]);
        if (err > max_quant_err) max_quant_err = err;
    }
    printf("量化最大误差: %.6f\n", max_quant_err);

    // ---- Step 2: GPU Dequantize Kernel ----
    printf("\n--- Step 2: GPU Dequantize Kernel ---\n");

    uint8_t* d_packed;
    float* d_scale;
    half* d_deq_out;

    CHECK_CUDA(cudaMalloc(&d_packed, K * (N / 2) * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_scale, K * num_groups_per_row * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_deq_out, K * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_packed, h_packed, K * (N / 2) * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale, K * num_groups_per_row * sizeof(float), cudaMemcpyHostToDevice));

    int total_packed = K * (N / 2);
    int threads = 256;
    int blocks_deq = (total_packed + threads - 1) / threads;
    dequantize_int4_kernel<<<blocks_deq, threads>>>(d_packed, d_scale, d_deq_out, K, N, group_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 验证 GPU dequant 结果
    half* h_deq_gpu = (half*)malloc(K * N * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_deq_gpu, d_deq_out, K * N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_gpu_err = 0.0f;
    for (int i = 0; i < K * N; i++) {
        float gpu_val = __half2float(h_deq_gpu[i]);
        float err = fabsf(h_dequant[i] - gpu_val);
        if (err > max_gpu_err) max_gpu_err = err;
    }
    printf("GPU vs CPU dequant 最大误差: %.6f (应为 ~0，FP16 舍入误差)\n", max_gpu_err);

    // ---- Step 3: W4A16 GEMM Kernel ----
    printf("\n--- Step 3: W4A16 GEMM Kernel ---\n");

    // 分配 A 和 C
    half* d_A;
    half* d_C_naive;
    half* d_C_tiled;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_naive, M * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_tiled, M * N * sizeof(half)));

    // 转换 A 到 FP16 并上传
    half* h_A_half = (half*)malloc(M * K * sizeof(half));
    for (int i = 0; i < M * K; i++) {
        h_A_half[i] = __float2half(h_A[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A_half, M * K * sizeof(half), cudaMemcpyHostToDevice));

    // 运行 naive kernel
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + 15) / 16, (M + 15) / 16);
    w4a16_gemm_naive<<<grid_naive, block_naive>>>(
        d_A, d_packed, d_scale, d_C_naive, M, N, K, group_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 运行 tiled kernel
    dim3 block_tiled(TILE_N, TILE_M);
    dim3 grid_tiled((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    w4a16_gemm_tiled<<<grid_tiled, block_tiled>>>(
        d_A, d_packed, d_scale, d_C_tiled, M, N, K, group_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // CPU 参考: A @ dequant(B)
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[m * K + k] * h_dequant[k * N + n];
            }
            h_C_ref[m * N + n] = sum;
        }
    }

    // 验证 naive kernel
    half* h_C_naive = (half*)malloc(M * N * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err_naive = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C_ref[i] - __half2float(h_C_naive[i]));
        if (err > max_err_naive) max_err_naive = err;
    }
    printf("Naive GEMM 最大误差 vs CPU: %.6f\n", max_err_naive);

    // 验证 tiled kernel
    half* h_C_tiled = (half*)malloc(M * N * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C_tiled, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_err_tiled = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C_ref[i] - __half2float(h_C_tiled[i]));
        if (err > max_err_tiled) max_err_tiled = err;
    }
    printf("Tiled GEMM 最大误差 vs CPU: %.6f\n", max_err_tiled);

    // ---- Benchmark (简单计时) ----
    printf("\n--- Benchmark ---\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    int n_iter = 100;

    // Naive
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; i++) {
        w4a16_gemm_naive<<<grid_naive, block_naive>>>(
            d_A, d_packed, d_scale, d_C_naive, M, N, K, group_size
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Naive  GEMM: %.4f ms (avg of %d runs)\n", elapsed / n_iter, n_iter);

    // Tiled
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; i++) {
        w4a16_gemm_tiled<<<grid_tiled, block_tiled>>>(
            d_A, d_packed, d_scale, d_C_tiled, M, N, K, group_size
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tiled  GEMM: %.4f ms (avg of %d runs)\n", elapsed / n_iter, n_iter);

    // 数据流总结
    printf("\n================================================================\n");
    printf("W4A16 GEMM 数据流总结:\n");
    printf("================================================================\n");
    printf("1. 权重存储: FP32 → INT4 per-group 量化 → packed (uint8)\n");
    printf("   [K, N] float → [K, N/2] uint8 + [K, N/gs] float scale\n");
    printf("   存储量: %.1f KB → %.1f KB (%.1fx 压缩)\n",
           K * N * 4.0f / 1024,
           (K * N / 2.0f + K * num_groups_per_row * 4.0f) / 1024,
           (K * N * 4.0f) / (K * N / 2.0f + K * num_groups_per_row * 4.0f));
    printf("\n");
    printf("2. GEMM kernel 内部:\n");
    printf("   a. 从 global mem 加载 packed INT4 (带宽省 4x)\n");
    printf("   b. 解包: uint8 → 两个 INT4\n");
    printf("   c. 符号扩展: unsigned 4-bit → signed\n");
    printf("   d. 反量化: INT4 * scale → FP16/FP32\n");
    printf("   e. FMA: acc += a_fp16 * b_fp16\n");
    printf("\n");
    printf("3. 生产级优化 (Marlin kernel):\n");
    printf("   - 用 Tensor Core MMA 指令做 FP16 matmul\n");
    printf("   - Async copy (cp.async) 从 global → shared memory\n");
    printf("   - Double buffering / software pipelining\n");
    printf("   - Register-level tiling\n");
    printf("   - Warp-level parallelism\n");
    printf("================================================================\n");

    // 清理
    free(h_weight); free(h_A); free(h_packed); free(h_scale); free(h_dequant);
    free(h_deq_gpu); free(h_A_half); free(h_C_ref); free(h_C_naive); free(h_C_tiled);
    cudaFree(d_packed); cudaFree(d_scale); cudaFree(d_deq_out);
    cudaFree(d_A); cudaFree(d_C_naive); cudaFree(d_C_tiled);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
