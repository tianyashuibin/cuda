"""
Step 6: Quantized Matrix Multiplication (Weight-Only INT8)
===========================================================
LLM 推理优化的核心：量化矩阵乘法

为什么需要量化：
  - LLM 推理是 memory-bandwidth bound（大部分时间在等数据从显存搬到计算单元）
  - 7B 模型 FP16 权重 = 14GB，INT8 = 7GB，INT4 = 3.5GB
  - 权重小了 → 加载快了 → 推理快了

Weight-Only Quantization:
  - 权重用 INT8 存储（省显存+带宽）
  - 激活保持 FP16（保精度）
  - 在 kernel 内部做 dequantize: w_fp16 = w_int8 * scale
  - Dequantize 和 matmul fused 在同一个 kernel（零额外开销）

对标方案：GPTQ, AWQ, SmoothQuant 等
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 量化工具函数
# ============================================================

def quantize_per_channel(weight: torch.Tensor):
    """
    Per-channel INT8 对称量化
    weight: [K, N] float16
    returns: (weight_int8, scale)
      weight_int8: [K, N] int8
      scale: [N] float16  (每列一个 scale factor)
    """
    # 对称量化：scale = max(abs(w)) / 127
    abs_max = weight.abs().amax(dim=0)  # [N]
    scale = abs_max / 127.0
    scale = scale.clamp(min=1e-8)  # 避免除零

    # 量化: w_int8 = round(w / scale)
    weight_int8 = torch.round(weight / scale[None, :]).to(torch.int8)

    return weight_int8, scale.to(torch.float16)


# ============================================================
# Triton Quantized MatMul Kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def quantized_matmul_kernel(
    # A: 激活 [M, K] float16
    # B: 量化权重 [K, N] int8
    # Scale: per-channel scale [N] float16
    # C: 输出 [M, N] float16
    A, B, Scale, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    C = A @ (B * Scale)
    其中 B 是 INT8，Scale 是 per-channel FP16
    Dequantize 在 kernel 内 fused 完成
    """
    # Grouped ordering (同 Step 3)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算偏移
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # K 维度循环累加
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K

        # 加载 A tile [BLOCK_M, BLOCK_K] float16
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)

        # 加载 B tile [BLOCK_K, BLOCK_N] int8
        # 关键：加载 INT8 数据，在寄存器中转换为 FP16
        b_int8 = tl.load(b_ptrs, mask=k_mask[:, None], other=0)
        b = b_int8.to(tl.float16)  # INT8 → FP16 (dequantize 的一部分)

        # 矩阵乘加
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ---- Fused Dequantize: 乘以 per-channel scale ----
    # 这一步和 matmul 融合在同一个 kernel，零额外 memory traffic
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    scale = tl.load(Scale + offs_cn, mask=offs_cn < N)
    c = accumulator.to(tl.float16) * scale[None, :]

    # 写回结果
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def quantized_matmul(
    activation: torch.Tensor,     # [M, K] float16
    weight_int8: torch.Tensor,    # [K, N] int8
    scale: torch.Tensor,          # [N] float16
) -> torch.Tensor:
    """Triton quantized matmul: output = activation @ (weight_int8 * scale)"""
    assert activation.is_cuda
    M, K = activation.shape
    K2, N = weight_int8.shape
    assert K == K2

    output = torch.empty((M, N), device=activation.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    quantized_matmul_kernel[grid](
        activation, weight_int8, scale, output,
        M, N, K,
        activation.stride(0), activation.stride(1),
        weight_int8.stride(0), weight_int8.stride(1),
        output.stride(0), output.stride(1),
    )
    return output


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    torch.manual_seed(0)
    configs = [
        # (M, K, N) — 模拟 LLM 推理场景
        (1,    4096, 4096),     # 单 token 推理 (decode)
        (32,   4096, 4096),     # 小 batch 推理
        (128,  4096, 4096),     # 中 batch
        (1,    4096, 11008),    # LLaMA-7B FFN up/gate projection
        (32,   4096, 11008),
        (128,  4096, 11008),
    ]

    print(f"{'M×K×N':>20} | {'FP16 (ms)':>10} | {'INT8 Triton (ms)':>16} | {'Speedup':>8} | {'Max Diff':>10}")
    print("-" * 78)

    for M, K, N in configs:
        # 原始 FP16 权重
        weight_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
        activation = torch.randn(M, K, device='cuda', dtype=torch.float16)

        # 量化
        weight_int8, scale = quantize_per_channel(weight_fp16)

        # 正确性验证
        # 参考：用 dequantized weight 做标准 matmul
        weight_deq = weight_int8.to(torch.float16) * scale[None, :]
        ref = torch.matmul(activation, weight_deq)
        out = quantized_matmul(activation, weight_int8, scale)
        max_diff = torch.max(torch.abs(ref.float() - out.float())).item()

        # 性能测试
        ms_fp16 = triton.testing.do_bench(lambda: torch.matmul(activation, weight_fp16))
        ms_int8 = triton.testing.do_bench(lambda: quantized_matmul(activation, weight_int8, scale))

        speedup = ms_fp16 / ms_int8
        shape_str = f"{M}×{K}×{N}"
        print(f"{shape_str:>20} | {ms_fp16:>10.4f} | {ms_int8:>16.4f} | {speedup:>7.2f}x | {max_diff:>10.2e}")

    print("\n注意：INT8 量化的加速主要来自减少 memory bandwidth，")
    print("在小 M (decode) 场景收益最大（memory-bound），大 M 收益递减（趋向 compute-bound）。")


if __name__ == "__main__":
    print("=" * 78)
    print("Step 6: Quantized MatMul (Weight-Only INT8)")
    print("=" * 78)

    # 基本功能测试
    print("\n--- 功能测试 ---")
    M, K, N = 64, 256, 128
    weight = torch.randn(K, N, device='cuda', dtype=torch.float16)
    activation = torch.randn(M, K, device='cuda', dtype=torch.float16)

    # 量化
    weight_int8, scale = quantize_per_channel(weight)
    print(f"原始权重: {weight.shape} ({weight.element_size() * weight.numel() / 1024:.1f} KB)")
    print(f"量化权重: {weight_int8.shape} ({weight_int8.element_size() * weight_int8.numel() / 1024:.1f} KB)")
    print(f"压缩率: {weight.element_size() / weight_int8.element_size():.0f}x")

    # 计算
    out = quantized_matmul(activation, weight_int8, scale)
    weight_deq = weight_int8.to(torch.float16) * scale[None, :]
    ref = torch.matmul(activation, weight_deq)
    print(f"Output shape: {out.shape}")
    print(f"Max diff vs dequantized reference: {torch.max(torch.abs(ref.float() - out.float())).item():.2e}")

    # Benchmark
    print("\n--- 性能对比 (FP16 cuBLAS vs INT8 Triton) ---")
    benchmark()
