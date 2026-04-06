#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Thread block size
// 可以尝试调整 BLOCK_SIZE 为 16/32
#define BLOCK_SIZE 32

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
// Returns transfer time and kernel time in milliseconds via output parameters
void MatMul(const Matrix A, const Matrix B, Matrix C,
            float *transfer_ms, float *kernel_ms)
{
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    // Allocate device memory
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

    // H2D transfer
    cudaEventRecord(e0);
    cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    // Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    cudaEventRecord(e2);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(e3);

    // D2H transfer
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

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    SetElement(Csub, row, col, Cvalue);
}

// CPU reference implementation for verification
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
    // Matrix dimensions must be multiples of BLOCK_SIZE
    int N = 1024; // Use 64x64 for a quick test; try 512 or 1024 for benchmarking

    // Allocate host matrices
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

    // Initialize A and B with simple values
    for (int i = 0; i < N * N; ++i) {
        A.elements[i] = (float)(i % 10) * 0.1f;
        B.elements[i] = (float)(i % 7)  * 0.1f;
    }

    // Run GPU matmul
    float transfer_ms = 0, kernel_ms = 0;
    MatMul(A, B, C_gpu, &transfer_ms, &kernel_ms);

    // Run CPU reference
    clock_t cpu_start = clock();
    MatMulCPU(A, B, C_cpu);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify results
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float err = fabsf(C_gpu.elements[i] - C_cpu.elements[i]);
        if (err > max_err) max_err = err;
    }

    printf("Matrix size: %dx%d\n", N, N);
    printf("GPU kernel time:   %.3f ms\n", kernel_ms);
    printf("GPU transfer time: %.3f ms (H2D + D2H)\n", transfer_ms);
    printf("GPU total time:    %.3f ms\n", kernel_ms + transfer_ms);
    printf("CPU time:          %.3f ms\n", cpu_ms);
    printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms);
    printf("Speedup (total):       %.1fx\n", cpu_ms / (kernel_ms + transfer_ms));
    printf("Max error (GPU vs CPU): %e\n", max_err);
    if (max_err < 1e-3f)
        printf("PASSED\n");
    else
        printf("FAILED\n");

    free(A.elements);
    free(B.elements);
    free(C_gpu.elements);
    free(C_cpu.elements);

    return 0;
}
