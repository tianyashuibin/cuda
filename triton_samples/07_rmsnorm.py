"""
Step 7: RMS Normalization (Forward + Backward)
================================================
LLaMA / Mistral / Gemma 等现代 LLM 使用 RMSNorm 替代 LayerNorm

RMSNorm vs LayerNorm:
  - LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b
  - RMSNorm:   y = x / sqrt(mean(x²) + eps) * w
  - 区别：没有 mean subtraction，没有 bias
  - 优势：少一次 reduction（不算 mean），参数少一半（无 bias），计算更快
  - 效果：实践中 RMSNorm 和 LayerNorm 精度相当，训练收敛速度接近

为什么 RMSNorm 适合 Triton 优化：
  - Memory-bound 算子：一行数据只做 O(D) 次计算，但需要两次全局读写
  - Fused kernel 将 rms → normalize → scale 合并，减少 global memory 访问
  - 推理时 batch size 小，launch overhead 占比大，fusion 收益明显
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Forward Kernel
# ============================================================

@triton.jit
def rmsnorm_fwd_kernel(
    X,        # 输入 [N, D]
    Y,        # 输出 [N, D]
    W,        # scale 参数 (gamma) [D]
    Rrms,     # 保存 1/rms（backward 需要）[N]
    stride,   # X 的行步长
    D,        # 每行的元素数（hidden size）
    eps,      # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm forward:
      rms = sqrt(mean(x²) + eps)
      y = (x / rms) * w
    """
    # 每个 program 处理一行
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # 加载一整行到寄存器
    x_ptr = X + row * stride
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 1. 计算 mean(x²)
    #    注意：这里没有减均值，这是 RMSNorm 和 LayerNorm 的核心区别
    sq_mean = tl.sum(x * x, axis=0) / D

    # 2. 计算 1/rms
    rrms = 1.0 / tl.sqrt(sq_mean + eps)

    # 3. 归一化 + Scale
    x_hat = x * rrms
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = x_hat * w

    # 写回结果
    y_ptr = Y + row * stride
    tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=mask)

    # 保存中间量（backward 需要）
    tl.store(Rrms + row, rrms)


# ============================================================
# Backward Kernel
# ============================================================

@triton.jit
def rmsnorm_bwd_kernel(
    DY,       # 上游梯度 [N, D]
    X,        # 输入 [N, D]
    W,        # scale 参数 [D]
    Rrms,     # forward 保存的 1/rms [N]
    DX,       # 输入梯度 [N, D]
    DW,       # scale 梯度 [D]（需要 atomic add）
    stride,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm backward 推导:

    Forward:  y = x * rrms * w,  其中 rrms = 1/sqrt(mean(x²) + eps)

    对 x 求导（链式法则）:
      令 s = mean(x²) + eps = (1/D) * sum(x²) + eps
      rrms = s^(-1/2)

      dy/dx_i = rrms * w_i + x_i * d(rrms)/dx_i * w_i
      d(rrms)/dx_i = -1/2 * s^(-3/2) * (2*x_i/D) = -rrms³ * x_i / D

      因此:
      dx_hat_i = dy_i * w_i  (先乘 scale)
      dx_i = rrms * (dx_hat_i - x_hat_i * mean(dx_hat_i * x_hat_i))
      其中 x_hat = x * rrms

    对 w 求导:
      dw_i = sum_over_rows(dy_i * x_hat_i)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # 加载数据
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rrms = tl.load(Rrms + row)

    # 计算 x_hat = x * rrms
    x_hat = x * rrms

    # dx_hat = dy * w
    dx_hat = dy * w

    # dx = rrms * (dx_hat - x_hat * mean(dx_hat * x_hat))
    c = tl.sum(dx_hat * x_hat, axis=0) / D
    dx = rrms * (dx_hat - x_hat * c)

    # 写回 dx
    tl.store(DX + row * stride + cols, dx.to(DX.dtype.element_ty), mask=mask)

    # dw 需要跨行累加
    tl.atomic_add(DW + cols, (dy * x_hat).to(DW.dtype.element_ty), mask=mask)


# ============================================================
# Python wrapper
# ============================================================

class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        assert x.is_cuda
        N, D = x.shape
        BLOCK_SIZE = triton.next_power_of_2(D)

        y = torch.empty_like(x)
        rrms = torch.empty(N, device=x.device, dtype=torch.float32)

        rmsnorm_fwd_kernel[(N,)](
            x, y, weight, rrms,
            x.stride(0), D, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight, rrms)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.D = D
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rrms = ctx.saved_tensors
        N, D = x.shape

        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)

        rmsnorm_bwd_kernel[(N,)](
            dy, x, weight, rrms,
            dx, dw,
            x.stride(0), D,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        return dx, dw.to(weight.dtype), None


def triton_rmsnorm(x, weight, eps=1e-6):
    """Triton RMSNorm (支持 autograd)"""
    return TritonRMSNorm.apply(x, weight, eps)


# ============================================================
# PyTorch Reference
# ============================================================

def pytorch_rmsnorm(x, weight, eps=1e-6):
    """PyTorch 参考实现"""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms * weight).to(x.dtype)


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    torch.manual_seed(0)
    configs = [
        (1024, 768),      # BERT-base hidden size
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),     # LLaMA-7B hidden size
        (8192, 4096),
        (1, 4096),        # 单 token decode
        (32, 4096),       # 小 batch decode
    ]

    print(f"{'Shape':>16} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)

    for N, D in configs:
        x = torch.randn(N, D, device='cuda', dtype=torch.float16)
        w = torch.randn(D, device='cuda', dtype=torch.float16)

        # 正确性验证
        ref = pytorch_rmsnorm(x, w)
        out = triton_rmsnorm(x, w)
        assert torch.allclose(ref.float(), out.float(), atol=1e-2, rtol=1e-3), \
            f"Forward 不一致! max diff={torch.max(torch.abs(ref.float() - out.float())).item()}"

        # 性能测试
        ms_torch = triton.testing.do_bench(lambda: pytorch_rmsnorm(x, w))
        ms_triton = triton.testing.do_bench(lambda: triton_rmsnorm(x, w))

        speedup = ms_torch / ms_triton
        shape_str = f"{N}x{D}"
        print(f"{shape_str:>16} | {ms_torch:>12.4f} | {ms_triton:>12.4f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 7: RMS Normalization")
    print("=" * 60)

    # Forward 测试
    print("\n--- Forward 功能测试 ---")
    N, D = 128, 768
    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    w = torch.ones(D, device='cuda', dtype=torch.float16)

    out = triton_rmsnorm(x, w)
    ref = pytorch_rmsnorm(x, w)
    print(f"Input: [{N}, {D}]")
    print(f"Max diff vs PyTorch: {torch.max(torch.abs(ref.float() - out.float())).item():.2e}")

    # 验证 RMSNorm 的特性：输出的 RMS ≈ 1（当 weight=1 时）
    rms_out = torch.sqrt(torch.mean(out.float() ** 2, dim=-1))
    print(f"Output RMS (should be ~1): mean={rms_out.mean().item():.4f}, std={rms_out.std().item():.4f}")

    # Backward 测试
    print("\n--- Backward 功能测试 ---")
    x = torch.randn(64, 256, device='cuda', dtype=torch.float32, requires_grad=True)
    w = torch.randn(256, device='cuda', dtype=torch.float32, requires_grad=True)
    dy = torch.randn(64, 256, device='cuda', dtype=torch.float32)

    # PyTorch reference backward (手动实现)
    ref_out = pytorch_rmsnorm(x, w)
    ref_out.backward(dy)
    ref_dx, ref_dw = x.grad.clone(), w.grad.clone()

    # Triton backward
    x.grad, w.grad = None, None
    tri_out = triton_rmsnorm(x, w)
    tri_out.backward(dy)

    print(f"dx max diff: {torch.max(torch.abs(ref_dx - x.grad)).item():.2e}")
    print(f"dw max diff: {torch.max(torch.abs(ref_dw - w.grad)).item():.2e}")

    # RMSNorm vs LayerNorm 对比说明
    print("\n--- RMSNorm vs LayerNorm ---")
    print("RMSNorm:   y = x / rms(x) * w          (无 mean, 无 bias)")
    print("LayerNorm: y = (x-mean) / std(x) * w + b (有 mean, 有 bias)")
    print("RMSNorm 少一次 reduction，参数少一半，LLaMA/Mistral/Gemma 均使用 RMSNorm")

    # Benchmark
    print("\n--- 性能对比 (Forward) ---")
    benchmark()
