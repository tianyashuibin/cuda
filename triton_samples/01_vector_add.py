"""
Step 1: Vector Add — Triton Hello World
========================================
对标 CUDA: cuda_samples/vector_add.cu

核心概念：
- @triton.jit 定义 kernel
- tl.program_id(0) ≈ CUDA blockIdx.x
- tl.arange(0, BLOCK_SIZE) ≈ CUDA threadIdx.x
- tl.load / tl.store ≈ CUDA global memory 读写
- mask 处理边界 ≈ CUDA if (idx < n)

关键区别：Triton 以 block 为编程单位，一次处理 BLOCK_SIZE 个元素
         CUDA 以 thread 为编程单位，每个线程处理 1 个元素
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,          # 输入向量 A 的指针
    b_ptr,          # 输入向量 B 的指针
    c_ptr,          # 输出向量 C 的指针
    n_elements,     # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program (block) 处理的元素数
):
    # ---- 等价于 CUDA: int blockIdx = blockIdx.x ----
    pid = tl.program_id(axis=0)

    # ---- 等价于 CUDA: int idx = blockIdx.x * blockDim.x + threadIdx.x ----
    # 但这里一次生成整个 block 的索引向量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ---- 等价于 CUDA: if (idx < n) ----
    mask = offsets < n_elements

    # ---- 等价于 CUDA: float a = A[idx], b = B[idx] ----
    # tl.load 加载整个 block 的数据（编译器自动处理向量化、coalescing）
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # ---- 等价于 CUDA: c[idx] = a[idx] + b[idx] ----
    c = a + b

    # ---- 等价于 CUDA: C[idx] = c ----
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton vector add wrapper"""
    assert a.shape == b.shape
    assert a.is_cuda and b.is_cuda
    c = torch.empty_like(a)
    n_elements = a.numel()

    # BLOCK_SIZE 是 Triton 的核心调优参数
    # 类比 CUDA: blockDim.x = 1024
    BLOCK_SIZE = 1024

    # grid: launch 多少个 program (block)
    # 类比 CUDA: gridDim.x = (n + blockSize - 1) / blockSize
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    # 类比 CUDA: vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N)
    vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return c


# ============================================================
# Benchmark: Triton vs PyTorch
# ============================================================

def benchmark():
    """
    对比 Triton kernel 和 PyTorch 原生实现的性能

    注意：vector_add 过于简单（纯 memory-bound，无 fusion 机会），
    PyTorch 的 a+b 已经是单个优化好的 CUDA kernel。
    这里的目的是学习 Triton 编程模型，不是追求超越 PyTorch。
    Triton 在数据量大时应接近 PyTorch（两者都受限于显存带宽）。
    """
    torch.manual_seed(0)
    sizes = [2**i for i in [16, 18, 20, 22, 24, 26]]

    print(f"{'N':>12} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Ratio':>8}")
    print("-" * 55)

    for n in sizes:
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)

        # 正确性验证
        c_torch = a + b
        c_triton = vector_add(a, b)
        assert torch.allclose(c_torch, c_triton), "结果不一致!"

        # 性能测试
        ms_torch = triton.testing.do_bench(lambda: a + b)
        ms_triton = triton.testing.do_bench(lambda: vector_add(a, b))

        ratio = ms_triton / ms_torch
        print(f"{n:>12,} | {ms_torch:>12.4f} | {ms_triton:>12.4f} | {ratio:>6.2f}x")

    print("\n* Ratio = Triton/PyTorch，越接近 1.0 越好，>1 表示 Triton 更慢")
    print("* vector_add 是 memory-bound 操作，两者最终都受限于显存带宽，差距应该很小")


if __name__ == "__main__":
    print("=" * 55)
    print("Step 1: Vector Add — Triton Hello World")
    print("=" * 55)

    # 基本功能测试
    print("\n--- 功能测试 ---")
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    c = vector_add(a, b)
    print(f"Input size: {a.shape}")
    print(f"First 5 results: {c[:5].tolist()}")
    print(f"Correct: {torch.allclose(a + b, c)}")

    # Benchmark
    print("\n--- 性能对比 ---")
    benchmark()
