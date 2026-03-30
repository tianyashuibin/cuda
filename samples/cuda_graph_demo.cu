// CUDA Graph Demo
//
// What is CUDA Graph?
//   Normally, every kernel launch is a separate CPU→GPU API call. Each call
//   has CPU overhead (~5~20 μs) for validation, command encoding, and
//   submission to the GPU command queue. For workloads with many small
//   kernels (like neural network inference), this CPU overhead dominates.
//
//   CUDA Graphs solve this by:
//   1. "Capture" a sequence of GPU operations into a graph (one-time cost)
//   2. "Replay" the entire graph with a single API call, skipping per-launch
//      CPU overhead entirely.
//
// This demo simulates a pipeline of 5 element-wise kernels applied to a
// vector, repeated REPEAT times. We compare:
//   A) Normal loop: REPEAT × 5 separate kernel launches
//   B) CUDA Graph:  capture once, then replay REPEAT times with 1 call each
//
// The speedup comes entirely from eliminating CPU launch overhead.
//
// Compile:
//   nvcc -O2 -o cuda_graph_demo cuda_graph_demo.cu -lm
// Run:
//   ./cuda_graph_demo

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N      (1 << 20)   // 1M elements per vector
#define REPEAT 1000        // how many times to replay the pipeline

// --- Simple pipeline kernels ---
// Each kernel does a lightweight element-wise operation.
// Individually trivial; the point is the *number of launches*.

__global__ void Scale(float *x, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= alpha;
}

__global__ void AddScalar(float *x, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += val;
}

__global__ void ReLU(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void Clip(float *x, float lo, float hi, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if      (x[i] < lo) x[i] = lo;
        else if (x[i] > hi) x[i] = hi;
    }
}

__global__ void Square(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * x[i];
}

// Launch one pass of the 5-kernel pipeline
void LaunchPipeline(float *d_x, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    Scale    <<<blocks, threads, 0, stream>>>(d_x, 0.9f,  n);
    AddScalar<<<blocks, threads, 0, stream>>>(d_x, 0.5f,  n);
    ReLU     <<<blocks, threads, 0, stream>>>(d_x,        n);
    Clip     <<<blocks, threads, 0, stream>>>(d_x, 0.0f, 10.0f, n);
    Square   <<<blocks, threads, 0, stream>>>(d_x,        n);
}

// Initialize host data
void InitData(float *h_x, int n)
{
    for (int i = 0; i < n; ++i)
        h_x[i] = (float)(i % 100) * 0.1f - 5.0f;  // range [-5, 5)
}

int main()
{
    printf("Vector size: %d elements\n", N);
    printf("Pipeline:    5 kernels × %d repeats = %d total launches\n\n",
           REPEAT, 5 * REPEAT);

    // Host buffers
    float *h_x      = (float*)malloc(N * sizeof(float));
    float *h_result_normal = (float*)malloc(N * sizeof(float));
    float *h_result_graph  = (float*)malloc(N * sizeof(float));

    // Device buffer
    float *d_x;
    cudaMalloc(&d_x, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ----------------------------------------------------------------
    // A) Normal execution: REPEAT × LaunchPipeline, each with 5 launches
    // ----------------------------------------------------------------
    InitData(h_x, N);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    LaunchPipeline(d_x, N, stream);
    cudaStreamSynchronize(stream);

    // Reset and time
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(e0, stream);
    for (int r = 0; r < REPEAT; ++r)
        LaunchPipeline(d_x, N, stream);
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);

    float normal_ms = 0;
    cudaEventElapsedTime(&normal_ms, e0, e1);
    cudaMemcpy(h_result_normal, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("[Normal]     %.3f ms  (%.2f μs per repeat)\n",
           normal_ms, normal_ms * 1000.0f / REPEAT);

    // ----------------------------------------------------------------
    // B) CUDA Graph: capture the pipeline once, replay REPEAT times
    // ----------------------------------------------------------------
    InitData(h_x, N);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // --- Capture phase ---
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    LaunchPipeline(d_x, N, stream);          // record one pass into the graph
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Warmup
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // Reset and time
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(e0, stream);
    for (int r = 0; r < REPEAT; ++r)
        cudaGraphLaunch(graphExec, stream);  // single API call per repeat
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);

    float graph_ms = 0;
    cudaEventElapsedTime(&graph_ms, e0, e1);
    cudaMemcpy(h_result_graph, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("[CUDA Graph] %.3f ms  (%.2f μs per repeat)\n",
           graph_ms, graph_ms * 1000.0f / REPEAT);
    printf("\nSpeedup: %.2fx\n\n", normal_ms / graph_ms);

    // ----------------------------------------------------------------
    // Verify: both runs should produce the same result
    // ----------------------------------------------------------------
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = fabsf(h_result_normal[i] - h_result_graph[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error between normal and graph: %e  %s\n",
           max_err, max_err < 1e-5f ? "PASSED" : "FAILED");

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
    free(h_x); free(h_result_normal); free(h_result_graph);

    return 0;
}
