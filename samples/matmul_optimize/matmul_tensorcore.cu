// Optimization 4: Tensor Core via WMMA (Warp Matrix Multiply Accumulate)
//
// Why it helps:
//   Tensor Cores are dedicated silicon for matrix multiply introduced in
//   Volta (sm_70). One warp can compute a full 16x16x16 FP16 matrix multiply
//   in a single instruction, far faster than scalar FP32/FP16 ALUs.
//   In practice Tensor Cores are what makes modern LLM training fast.
//
// How WMMA works:
//   - The unit of work is a "fragment": a 16x16 tile distributed across 32
//     threads in a warp (each thread holds a few elements).
//   - wmma::load_matrix_sync  — load a tile from memory into a fragment
//   - wmma::mma_sync          — multiply two fragments, accumulate into a third
//   - wmma::store_matrix_sync — write accumulator fragment back to memory
//
// This implementation:
//   - Input A, B: FP16
//   - Accumulator C: FP32 (higher precision for the sum)
//   - Each block = 1 warp (32 threads), computes one 16x16 output tile
//   - Grid = (N/16, N/16)
//
// Compile:
//   nvcc -O2 -arch=sm_70 -o matmul_tensorcore matmul_tensorcore.cu -lm
//   (requires Volta or newer: V100, T4, A100, H100, RTX 20xx/30xx/40xx)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <time.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each block is one warp (32 threads), computes one WMMA_M x WMMA_N output tile.
// A and B are FP16, C accumulates in FP32.
__global__ void MatMulTensorCore(const __half *A, const __half *B, float *C, int N)
{
    // This block's output tile position in the grid
    int warpRow = blockIdx.y;
    int warpCol = blockIdx.x;

    // Declare fragments
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                   c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Iterate over K dimension in WMMA_K steps
    for (int k = 0; k < N; k += WMMA_K) {
        // Pointer to the start of the 16x16 tile in A and B
        const __half *tileA = A + warpRow * WMMA_M * N + k;
        const __half *tileB = B + k * N + warpCol * WMMA_N;

        wmma::load_matrix_sync(a_frag, tileA, N);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store FP32 result
    float *tileC = C + warpRow * WMMA_M * N + warpCol * WMMA_N;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

void MatMulTC(const float *h_A, const float *h_B, float *h_C, int N,
              float *transfer_ms, float *kernel_ms)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    size_t countN  = (size_t)N * N;
    size_t sizeF16 = countN * sizeof(__half);
    size_t sizeF32 = countN * sizeof(float);

    // Convert host float → half
    __half *h_A_half = (__half*)malloc(sizeF16);
    __half *h_B_half = (__half*)malloc(sizeF16);
    for (size_t i = 0; i < countN; ++i) {
        h_A_half[i] = __float2half(h_A[i]);
        h_B_half[i] = __float2half(h_B[i]);
    }

    __half *d_A, *d_B;
    float  *d_C;
    cudaMalloc(&d_A, sizeF16);
    cudaMalloc(&d_B, sizeF16);
    cudaMalloc(&d_C, sizeF32);

    cudaEventRecord(e0);
    cudaMemcpy(d_A, h_A_half, sizeF16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half, sizeF16, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    // One warp (32 threads) per block, one block per 16x16 output tile
    dim3 dimBlock(32, 1);
    dim3 dimGrid(N / WMMA_N, N / WMMA_M);
    cudaEventRecord(e2);
    MatMulTensorCore<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e3);

    cudaEventRecord(e4);
    cudaMemcpy(h_C, d_C, sizeF32, cudaMemcpyDeviceToHost);
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
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A_half); free(h_B_half);
}

void MatMulCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main()
{
    int N = 1024; // must be multiple of 16
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A     = (float*)malloc(bytes);
    float *B     = (float*)malloc(bytes);
    float *C_gpu = (float*)malloc(bytes);
    float *C_cpu = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)(i % 10) * 0.1f;
        B[i] = (float)(i % 7)  * 0.1f;
    }

    float transfer_ms = 0, kernel_ms = 0;
    MatMulTC(A, B, C_gpu, N, &transfer_ms, &kernel_ms);

    clock_t cpu_start = clock();
    MatMulCPU(A, B, C_cpu, N);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // FP16 input → relaxed error threshold
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("[Tensor Core WMMA] Matrix size: %dx%d\n", N, N);
    printf("GPU kernel time:   %.3f ms\n", kernel_ms);
    printf("GPU transfer time: %.3f ms\n", transfer_ms);
    printf("GPU total time:    %.3f ms\n", kernel_ms + transfer_ms);
    printf("CPU time:          %.3f ms\n", cpu_ms);
    printf("Speedup (kernel):  %.1fx\n", cpu_ms / kernel_ms);
    printf("Max error: %e  %s\n", max_err, max_err < 0.1f ? "PASSED" : "FAILED");

    free(A); free(B); free(C_gpu); free(C_cpu);
    return 0;
}
