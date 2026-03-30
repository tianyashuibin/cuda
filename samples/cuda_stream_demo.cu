// CUDA Stream Demo
//
// What is a CUDA Stream?
//   A stream is a sequence of GPU operations that execute in order.
//   Operations in *different* streams can overlap (run concurrently):
//     - H2D transfer on stream 1  can overlap with  kernel on stream 2
//     - kernel on stream 1        can overlap with  D2H transfer on stream 2
//   This hides memory transfer latency behind computation.
//
// This demo processes a large array in chunks and compares:
//   A) No overlap (default stream): H2D → kernel → D2H  for each chunk,
//      all sequential.
//   B) Pipelined streams: N streams interleaved so transfers and compute
//      overlap like a CPU pipeline:
//
//      Stream 0: [H2D_0] [kernel_0] [D2H_0]
//      Stream 1:         [H2D_1] [kernel_1] [D2H_1]
//      Stream 2:                 [H2D_2] [kernel_2] [D2H_2]
//                ─────────────────────────────────────────▶ time
//
// Requirements:
//   Async memcpy REQUIRES pinned (page-locked) host memory.
//   cudaMallocHost() allocates pinned memory; cudaMemcpyAsync() uses DMA
//   to transfer without CPU involvement, enabling true overlap.
//
// Compile:
//   nvcc -O2 -o cuda_stream_demo cuda_stream_demo.cu -lm
// Run:
//   ./cuda_stream_demo

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TOTAL_N     (1 << 24)   // 16M elements total
#define NUM_STREAMS 4           // number of streams for pipelined version
#define CHUNK_N     (TOTAL_N / NUM_STREAMS)  // elements per chunk

// A moderately heavy kernel so compute time is visible
__global__ void ProcessKernel(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Simulate non-trivial computation (a few iterations of math)
    float val = in[i];
    for (int iter = 0; iter < 20; ++iter)
        val = sinf(val) * cosf(val) + 1.0f;
    out[i] = val;
}

void launchKernel(const float *d_in, float *d_out, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    ProcessKernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

int main()
{
    printf("Total elements: %d  (%.1f MB per buffer)\n",
           TOTAL_N, TOTAL_N * sizeof(float) / 1e6f);
    printf("Chunk size:     %d  (%d chunks)\n\n", CHUNK_N, NUM_STREAMS);

    // Pinned host memory — required for async (overlapping) transfers
    float *h_in, *h_out_seq, *h_out_stream;
    cudaMallocHost(&h_in,         TOTAL_N * sizeof(float));
    cudaMallocHost(&h_out_seq,    TOTAL_N * sizeof(float));
    cudaMallocHost(&h_out_stream, TOTAL_N * sizeof(float));

    for (int i = 0; i < TOTAL_N; ++i)
        h_in[i] = (float)(i % 1000) * 0.001f;

    // Device buffers — one pair per stream for pipelining
    float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaMalloc(&d_in[s],  CHUNK_N * sizeof(float));
        cudaMalloc(&d_out[s], CHUNK_N * sizeof(float));
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ----------------------------------------------------------------
    // A) Sequential (default stream, no overlap)
    //    H2D → kernel → D2H  for each chunk, one after another.
    // ----------------------------------------------------------------
    cudaEventRecord(e0);
    for (int s = 0; s < NUM_STREAMS; ++s) {
        int offset = s * CHUNK_N;
        cudaMemcpy(d_in[s], h_in + offset,
                   CHUNK_N * sizeof(float), cudaMemcpyHostToDevice);
        launchKernel(d_in[s], d_out[s], CHUNK_N, 0);
        cudaMemcpy(h_out_seq + offset, d_out[s],
                   CHUNK_N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float seq_ms = 0;
    cudaEventElapsedTime(&seq_ms, e0, e1);
    printf("[Sequential]    %.3f ms\n", seq_ms);

    // ----------------------------------------------------------------
    // B) Pipelined streams (overlap H2D / kernel / D2H across chunks)
    //    Each chunk gets its own stream; all streams are issued before
    //    waiting, letting the GPU overlap transfers and computation.
    // ----------------------------------------------------------------
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        cudaStreamCreate(&streams[s]);

    cudaEventRecord(e0);
    // Issue all operations across all streams without waiting in between
    for (int s = 0; s < NUM_STREAMS; ++s) {
        int offset = s * CHUNK_N;
        cudaMemcpyAsync(d_in[s], h_in + offset,
                        CHUNK_N * sizeof(float),
                        cudaMemcpyHostToDevice, streams[s]);
        launchKernel(d_in[s], d_out[s], CHUNK_N, streams[s]);
        cudaMemcpyAsync(h_out_stream + offset, d_out[s],
                        CHUNK_N * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
    }
    // Wait for all streams to finish
    for (int s = 0; s < NUM_STREAMS; ++s)
        cudaStreamSynchronize(streams[s]);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float stream_ms = 0;
    cudaEventElapsedTime(&stream_ms, e0, e1);
    printf("[Pipelined streams] %.3f ms\n", stream_ms);
    printf("\nSpeedup: %.2fx\n\n", seq_ms / stream_ms);

    // ----------------------------------------------------------------
    // Verify results are identical
    // ----------------------------------------------------------------
    float max_err = 0.0f;
    for (int i = 0; i < TOTAL_N; ++i) {
        float err = fabsf(h_out_seq[i] - h_out_stream[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error (sequential vs streams): %e  %s\n",
           max_err, max_err < 1e-5f ? "PASSED" : "FAILED");

    // Cleanup
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_in[s]);
        cudaFree(d_out[s]);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out_seq);
    cudaFreeHost(h_out_stream);

    return 0;
}
