// Optimization 1: Larger BLOCK_SIZE (32 instead of 16)
//
// Why it helps:
//   Each thread block loads BLOCK_SIZE^2 elements into shared memory and
//   reuses them BLOCK_SIZE times. Larger BLOCK_SIZE means better reuse ratio
//   and higher occupancy of shared memory bandwidth.
//   BLOCK_SIZE=32 → 32x32x4x2 = 8KB shared mem per block (within limits).
//   BLOCK_SIZE=16 → 16x16x4x2 = 2KB — leaves a lot of shared mem unused.
//
// Compile:
//   nvcc -O2 -o matmul_blocksize32 matmul_blocksize32.cu -lm
//
// Note: BLOCK_SIZE=32 means 32*32=1024 threads per block, which is the
//       maximum allowed by most CUDA GPUs.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 32

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub  = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

void MatMul(const Matrix A, const Matrix B, Matrix C,
            float *transfer_ms, float *kernel_ms)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size_t sizeA = A.width * A.height * sizeof(float);
    size_t sizeB = B.width * B.height * sizeof(float);
    size_t sizeC = C.width * C.height * sizeof(float);
    cudaMalloc(&d_A.elements, sizeA);
    cudaMalloc(&d_B.elements, sizeB);
    cudaMalloc(&d_C.elements, sizeC);

    cudaEventRecord(e0);
    cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / BLOCK_SIZE, A.height / BLOCK_SIZE);
    cudaEventRecord(e2);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(e3);

    cudaEventRecord(e4);
    cudaMemcpy(C.elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost);
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
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void MatMulCPU(const Matrix A, const Matrix B, Matrix C)
{
    for (int i = 0; i < A.height; ++i)
        for (int j = 0; j < B.width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.width; ++k)
                sum += A.elements[i * A.stride + k] * B.elements[k * B.stride + j];
            C.elements[i * C.stride + j] = sum;
        }
}

int main()
{
    int N = 1024; // must be multiple of BLOCK_SIZE (32)

    Matrix A, B, C_gpu, C_cpu;
    A.width = A.height = A.stride = N;
    B.width = B.height = B.stride = N;
    C_gpu.width = C_gpu.height = C_gpu.stride = N;
    C_cpu.width = C_cpu.height = C_cpu.stride = N;

    size_t bytes = N * N * sizeof(float);
    A.elements     = (float*)malloc(bytes);
    B.elements     = (float*)malloc(bytes);
    C_gpu.elements = (float*)malloc(bytes);
    C_cpu.elements = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        A.elements[i] = (float)(i % 10) * 0.1f;
        B.elements[i] = (float)(i % 7)  * 0.1f;
    }

    float transfer_ms = 0, kernel_ms = 0;
    MatMul(A, B, C_gpu, &transfer_ms, &kernel_ms);

    clock_t cpu_start = clock();
    MatMulCPU(A, B, C_cpu);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float err = fabsf(C_gpu.elements[i] - C_cpu.elements[i]);
        if (err > max_err) max_err = err;
    }

    printf("[BLOCK_SIZE=32] Matrix size: %dx%d\n", N, N);
    printf("GPU kernel time:   %.3f ms\n", kernel_ms);
    printf("GPU transfer time: %.3f ms\n", transfer_ms);
    printf("GPU total time:    %.3f ms\n", kernel_ms + transfer_ms);
    printf("CPU time:          %.3f ms\n", cpu_ms);
    printf("Speedup (kernel):  %.1fx\n", cpu_ms / kernel_ms);
    printf("Max error: %e  %s\n", max_err, max_err < 1e-3f ? "PASSED" : "FAILED");

    free(A.elements); free(B.elements);
    free(C_gpu.elements); free(C_cpu.elements);
    return 0;
}
