"""
Step 2: Fused Softmax
=====================
核心概念：Kernel Fusion — 把多个操作融合到一个 kernel 里

PyTorch naive softmax 执行流程（每步都是一次 kernel launch + global memory 读写）：
  1. max_val = x.max(dim=-1)           → kernel launch + 写 global memory
  2. x_shifted = x - max_val           → kernel launch + 读/写 global memory
  3. exp_x = x_shifted.exp()           → kernel launch + 读/写 global memory
  4. sum_val = exp_x.sum(dim=-1)       → kernel launch + 写 global memory
  5. result = exp_x / sum_val          → kernel launch + 读/写 global memory
  总计：5 次 kernel launch，大量 global memory traffic

Triton fused softmax（单次 kernel launch，数据留在寄存器/shared memory）：
  1. 一次 tl.load 加载整行
  2. 在寄存器中完成 max → sub → exp → sum → div
  3. 一次 tl.store 写回结果
  总计：1 次 kernel launch，最少的 memory traffic

这就是为什么 Triton 适合写 fused kernel：
CUDA 实现 fusion 需要手动管理 shared memory、同步等，代码量大
Triton 只需要把操作写在同一个 kernel 函数里，编译器处理其余一切
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,              # 每行的列数
    input_row_stride,    # 输入矩阵行步长
    output_row_stride,   # 输出矩阵行步长
    BLOCK_SIZE: tl.constexpr,  # 必须 >= n_cols，每个 program 处理一整行
):
    # 每个 program 处理一行
    row_idx = tl.program_id(0)

    # 计算这一行的起始地址
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ---- 加载整行数据到寄存器 ----
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # ---- 以下全部在寄存器中完成，零 global memory 访问 ----

    # 1. 数值稳定性：减去最大值（防止 exp 溢出）
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # 2. 指数
    numerator = tl.exp(row)

    # 3. 归一化
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # ---- 写回结果 ----
    out_row_start = output_ptr + row_idx * output_row_stride
    tl.store(out_row_start + col_offsets, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton fused softmax (沿最后一维)"""
    assert x.is_cuda
    n_rows, n_cols = x.shape

    # BLOCK_SIZE 必须是 2 的幂，且 >= n_cols（一个 program 处理完整一行）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    output = torch.empty_like(x)
    grid = (n_rows,)  # 每行一个 program

    softmax_kernel[grid](
        x, output,
        n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ============================================================
# Benchmark: Triton vs PyTorch
# ============================================================

def benchmark():
    """
    对比 Triton fused softmax 和 PyTorch 实现的性能

    注意：torch.softmax 底层调用 cuDNN 高度优化的实现。
    Triton 版本的优势在于「可定制的 fusion」—— 如果你需要
    softmax + 其他操作（如 dropout、mask）融合在一起，
    Triton 可以轻松做到，而 cuDNN 只提供固定的 softmax。
    """
    torch.manual_seed(0)
    configs = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 4096),
        (4096, 8192),
    ]

    print(f"{'Shape':>16} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Ratio':>8}")
    print("-" * 60)

    for rows, cols in configs:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)

        # 正确性验证
        ref = torch.softmax(x, dim=-1)
        out = softmax(x)
        assert torch.allclose(ref, out, atol=1e-5), f"结果不一致! max diff={torch.max(torch.abs(ref-out))}"

        # 性能测试
        ms_torch = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
        ms_triton = triton.testing.do_bench(lambda: softmax(x))

        ratio = ms_triton / ms_torch
        shape_str = f"{rows}x{cols}"
        print(f"{shape_str:>16} | {ms_torch:>12.4f} | {ms_triton:>12.4f} | {ratio:>6.2f}x")

    print("\n* Ratio = Triton/PyTorch，<1 表示 Triton 更快")
    print("* PyTorch 调用 cuDNN 优化实现，Triton 的价值在于可定制 fusion")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Fused Softmax")
    print("=" * 60)

    # 基本功能测试
    print("\n--- 功能测试 ---")
    x = torch.randn(4, 8, device='cuda')
    out = softmax(x)
    ref = torch.softmax(x, dim=-1)
    print(f"Input shape: {x.shape}")
    print(f"Output row sum (should be ~1.0): {out.sum(dim=-1).tolist()}")
    print(f"Max diff vs PyTorch: {torch.max(torch.abs(out - ref)).item():.2e}")

    # Benchmark
    print("\n--- 性能对比 ---")
    benchmark()
