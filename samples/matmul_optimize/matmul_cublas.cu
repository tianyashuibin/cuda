// Optimization 3: cuBLAS
//
// Why it helps:
//   cuBLAS is NVIDIA's official linear algebra library. Their engineers spend
//   years hand-tuning assembly-level kernels for every GPU architecture,
//   using techniques like register blocking, vectorized loads, async copies,
//   and pipeline overlap. The result typically beats hand-written kernels by
//   3~10x on large matrices.
//
// Key detail — row-major vs column-major:
//   cuBLAS assumes column-major storage (Fortran convention).
//   Our matrices are row-major (C convention).
//   Trick: C = A*B in row-major  ≡  C^T = B^T * A^T in column-major.
//   So we swap A and B in the cuBLAS call and pass them as-is;
//   cuBLAS interprets each row-major matrix as its transpose, and the
//   result stored in column-major is exactly our row-major C.
//
// Compile:
//   nvcc -O2 -o matmul_cublas matmul_cublas.cu -lcublas -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

void MatMulCuBLAS(const float *h_A, const float *h_B, float *h_C, int N,
                  float *transfer_ms, float *kernel_ms)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // H2D
    cudaEventRecord(e0);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Compute C = A * B for row-major matrices using the transpose trick:
    //   cublasSgemm computes: C_col = alpha * op(A_col) * op(B_col) + beta * C_col
    //   Passing row-major B as "A" and row-major A as "B" makes cuBLAS compute:
    //   C_col = B^T * A^T = (A*B)^T, which when read as row-major gives A*B.
    const float alpha = 1.0f, beta = 0.0f;

    // Warmup: first call triggers algorithm selection and workspace allocation,
    // which would otherwise inflate the measured kernel time.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(e2);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // no transpose on either arg
                N, N, N,                    // m, n, k
                &alpha,
                d_B, N,                    // "A" in cuBLAS = our B (row-major)
                d_A, N,                    // "B" in cuBLAS = our A (row-major)
                &beta,
                d_C, N);                   // result
    cudaEventRecord(e3);

    // D2H
    cudaEventRecord(e4);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, e0, e1);
    cudaEventElapsedTime(&kern_ms, e2, e3);
    cudaEventElapsedTime(&d2h_ms, e4, e5);
    *transfer_ms = h2d_ms + d2h_ms;
    *kernel_ms   = kern_ms;

    cublasDestroy(handle);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
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
    MatMulCuBLAS(A, B, C_gpu, N, &transfer_ms, &kernel_ms);

    clock_t cpu_start = clock();
    MatMulCPU(A, B, C_cpu, N);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("[cuBLAS] Matrix size: %dx%d\n", N, N);
    printf("GPU kernel time:   %.3f ms\n", kernel_ms);
    printf("GPU transfer time: %.3f ms\n", transfer_ms);
    printf("GPU total time:    %.3f ms\n", kernel_ms + transfer_ms);
    printf("CPU time:          %.3f ms\n", cpu_ms);
    printf("Speedup (kernel):  %.1fx\n", cpu_ms / kernel_ms);
    printf("Max error: %e  %s\n", max_err, max_err < 1e-3f ? "PASSED" : "FAILED");

    free(A); free(B); free(C_gpu); free(C_cpu);
    return 0;
}
