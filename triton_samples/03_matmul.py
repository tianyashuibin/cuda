"""
Step 3: Matrix Multiplication (GEMM)
=====================================
对标 CUDA: cuda_samples/matmul_optimize/ 整个系列

Triton 用 ~80 行代码覆盖了你在 CUDA 中 7 个 step 的大部分优化：
  - shared memory tiling     → tl.load 循环（编译器自动用 shared memory）
  - 寄存器 tiling (TM×TN)    → BLOCK_SIZE_M/N/K 参数控制
  - bank conflict             → 编译器自动处理
  - double buffering          → 编译器自动 prefetch
  - float4 向量化             → 编译器自动选择
  - Tensor Core               → tl.dot() 自动使用

核心编程模型：
  - 2D grid：每个 program 计算 C 的一个 BLOCK_M × BLOCK_N 子块
  - K 维度循环：每次加载 BLOCK_K 宽的 tile，用 tl.dot 做矩阵乘加
  - L2 优化：grouped ordering 提高 cache 命中率
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Auto-tuning: Triton 自动搜索最优配置
# 对比 CUDA：手动改 BLOCK_SIZE → 重编译 → 跑 benchmark → 循环
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # 矩阵指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 行步长（列主序时 = 列数）
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 编译期常量（auto-tune 搜索的参数）
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    C[M, N] = A[M, K] @ B[K, N]
    每个 program 计算 C 的一个 BLOCK_SIZE_M × BLOCK_SIZE_N 子块
    """
    # ---- Program ID 和 Grouped Ordering ----
    # Grouped ordering (L2 cache 优化):
    # 不按简单的行优先遍历，而是把相邻的 program 分组
    # 让它们访问 A/B 的相近区域，提高 L2 cache 命中率
    #
    # 类比 CUDA: 通常需要手动 swizzle blockIdx 来优化 L2
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 计算 A/B 的 tile 起始偏移 ----
    # A 的 tile: [pid_m * BLOCK_M : (pid_m+1) * BLOCK_M, 0 : BLOCK_K]
    # B 的 tile: [0 : BLOCK_K, pid_n * BLOCK_N : (pid_n+1) * BLOCK_N]
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 指针：a_ptrs[i, j] = a_ptr + i * stride_am + j * stride_ak
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ---- K 维度循环累加 ----
    # 类比 CUDA matmul_optimize:
    #   - 这里的循环 = shared memory tiling 的 tile 循环
    #   - tl.dot = 寄存器 tiling 的内层计算（编译器自动做寄存器分配）
    #   - 编译器自动做 double buffering / prefetch
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A/B 的一个 tile（编译器自动使用 shared memory）
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # 矩阵乘加（编译器自动使用 Tensor Core）
        accumulator = tl.dot(a, b, accumulator)

        # 移动到下一个 K tile
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # ---- 写回结果 ----
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton matmul wrapper: C = A @ B"""
    assert a.shape[1] == b.shape[0], f"维度不匹配: {a.shape} @ {b.shape}"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D grid: 总 program 数 = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ============================================================
# Benchmark: Triton vs PyTorch (cuBLAS) vs CUDA 手写
# ============================================================

def benchmark():
    """
    对比 Triton GEMM 和 cuBLAS 性能

    cuBLAS 是数十年优化的 GEMM 库，Triton 目标是接近它（80-95%）。
    重点看 TFLOPS 和 Triton/cuBLAS 比值。
    """
    torch.manual_seed(0)
    sizes = [512, 1024, 2048, 4096]

    print(f"{'M=N=K':>8} | {'cuBLAS (ms)':>12} | {'Triton (ms)':>12} | {'TFLOPS (Triton)':>15} | {'Triton/cuBLAS':>13}")
    print("-" * 75)

    for size in sizes:
        M = N = K = size
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        # 正确性验证
        ref = torch.matmul(a, b)
        out = matmul(a, b)
        assert torch.allclose(ref, out, atol=1e-1, rtol=1e-2), \
            f"结果不一致! max diff={torch.max(torch.abs(ref.float() - out.float())).item()}"

        # 性能测试
        ms_cublas = triton.testing.do_bench(lambda: torch.matmul(a, b))
        ms_triton = triton.testing.do_bench(lambda: matmul(a, b))

        # 计算 TFLOPS: 2*M*N*K / time / 1e12
        flops = 2 * M * N * K
        tflops_triton = flops / (ms_triton * 1e-3) / 1e12
        pct = ms_cublas / ms_triton * 100  # 达到 cuBLAS 的百分比

        print(f"{size:>8} | {ms_cublas:>12.4f} | {ms_triton:>12.4f} | {tflops_triton:>14.2f}T | {pct:>11.1f}%")

    print("\n* Triton/cuBLAS 表示达到 cuBLAS 性能的百分比，>80% 就不错")
    print("* cuBLAS 经过数十年优化，Triton 用 ~80 行 Python 达到这个比例已经很强")


if __name__ == "__main__":
    print("=" * 75)
    print("Step 3: Matrix Multiplication (GEMM)")
    print("=" * 75)

    # 基本功能测试
    print("\n--- 功能测试 ---")
    a = torch.randn(64, 128, device='cuda', dtype=torch.float16)
    b = torch.randn(128, 32, device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    ref = torch.matmul(a, b)
    print(f"A: {a.shape}, B: {b.shape} → C: {c.shape}")
    print(f"Max diff vs cuBLAS: {torch.max(torch.abs(ref.float() - c.float())).item():.2e}")

    # Benchmark
    print("\n--- 性能对比 (FP16) ---")
    benchmark()
