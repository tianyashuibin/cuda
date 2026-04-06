// Optimization 2: FP16 (half precision) computation
//
// Why it helps:
//   FP16 values are 2 bytes vs 4 bytes for FP32, so:
//   - Shared memory holds 2x more data per block
//   - Memory bandwidth effectively doubles
//   - Modern GPUs have dedicated FP16 ALUs with 2x throughput vs FP32
//
// Trade-off:
//   FP16 has lower precision (3-4 decimal digits vs 7 for FP32).
//   Acceptable for deep learning inference, may not suit scientific computing.
//
// Compile:
//   nvcc -O2 -arch=sm_70 -o matmul_fp16 matmul_fp16.cu -lm
//   (requires compute capability 7.0+ for full FP16 ALU support;
//    sm_60/sm_61 Pascal also supports __half but with less throughput)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <time.h>

#define BLOCK_SIZE 16

// Matrix struct using half precision
typedef struct {
    int width;
    int height;
    int stride;
    __half* elements;
} MatrixFP16;

__device__ __half GetElement(const MatrixFP16 A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(MatrixFP16 A, int row, int col, __half value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ MatrixFP16 GetSubMatrix(MatrixFP16 A, int row, int col)
{
    MatrixFP16 Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernelFP16(MatrixFP16 A, MatrixFP16 B, MatrixFP16 C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    MatrixFP16 Csub = GetSubMatrix(C, blockRow, blockCol);

    // Accumulate in FP32 to reduce rounding error, convert to FP16 at the end
    float Cvalue = 0.0f;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        MatrixFP16 Asub = GetSubMatrix(A, blockRow, m);
        MatrixFP16 Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ __half As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ __half Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += __half2float(As[row][e]) * __half2float(Bs[e][col]);
        __syncthreads();
    }

    SetElement(Csub, row, col, __float2half(Cvalue));
}

void MatMulFP16(const float *h_A, const float *h_B, float *h_C, int N,
                float *transfer_ms, float *kernel_ms)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    size_t countN  = (size_t)N * N;
    size_t sizeF32 = countN * sizeof(float);
    size_t sizeF16 = countN * sizeof(__half);

    // Convert host float → half before upload
    __half *h_A_half = (__half*)malloc(sizeF16);
    __half *h_B_half = (__half*)malloc(sizeF16);
    __half *h_C_half = (__half*)malloc(sizeF16);
    for (size_t i = 0; i < countN; ++i) {
        h_A_half[i] = __float2half(h_A[i]);
        h_B_half[i] = __float2half(h_B[i]);
    }

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeF16);
    cudaMalloc(&d_B, sizeF16);
    cudaMalloc(&d_C, sizeF16);

    cudaEventRecord(e0);
    cudaMemcpy(d_A, h_A_half, sizeF16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half, sizeF16, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    MatrixFP16 md_A = {N, N, N, d_A};
    MatrixFP16 md_B = {N, N, N, d_B};
    MatrixFP16 md_C = {N, N, N, d_C};

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    cudaEventRecord(e2);
    MatMulKernelFP16<<<dimGrid, dimBlock>>>(md_A, md_B, md_C);
    cudaEventRecord(e3);

    cudaEventRecord(e4);
    cudaMemcpy(h_C_half, d_C, sizeF16, cudaMemcpyDeviceToHost);
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    // Convert half → float for verification
    for (size_t i = 0; i < countN; ++i)
        h_C[i] = __half2float(h_C_half[i]);

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
    free(h_A_half); free(h_B_half); free(h_C_half);
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
    int N = 1024;
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
    MatMulFP16(A, B, C_gpu, N, &transfer_ms, &kernel_ms);

    clock_t cpu_start = clock();
    MatMulCPU(A, B, C_cpu, N);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // FP16 has lower precision; use a relaxed threshold
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("[FP16] Matrix size: %dx%d\n", N, N);
    printf("GPU kernel time:   %.3f ms\n", kernel_ms);
    printf("GPU transfer time: %.3f ms\n", transfer_ms);
    printf("GPU total time:    %.3f ms\n", kernel_ms + transfer_ms);
    printf("CPU time:          %.3f ms\n", cpu_ms);
    printf("Speedup (kernel):  %.1fx\n", cpu_ms / kernel_ms);
    printf("Max error: %e  %s\n", max_err, max_err < 0.1f ? "PASSED" : "FAILED");

    free(A); free(B); free(C_gpu); free(C_cpu);
    return 0;
}
