"""
Step 5: Layer Normalization (Forward + Backward)
=================================================
LLM 推理中频率最高的算子之一

为什么 LayerNorm 适合 Triton 优化：
  - 计算简单（mean, var, normalize, scale, bias）但 PyTorch 实现需要多次 kernel launch
  - Memory-bound 算子：fusion 的收益主要来自减少 global memory traffic
  - 推理时 batch size 通常很小，这类轻量算子的 launch overhead 占比大

实现包括：
  - Forward: fused mean → var → normalize → scale → bias
  - Backward: fused 梯度计算
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Forward Kernel
# ============================================================

@triton.jit
def layernorm_fwd_kernel(
    X,        # 输入 [N, D]
    Y,        # 输出 [N, D]
    W,        # scale 参数 (gamma) [D]
    B,        # bias 参数 (beta)  [D]
    Mean,     # 保存均值（backward 需要）[N]
    Rstd,     # 保存 1/std（backward 需要）[N]
    stride,   # X 的行步长
    N,        # 行数
    D,        # 每行的元素数（hidden size）
    eps,      # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一行
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # 加载一整行到寄存器
    x_ptr = X + row * stride
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- 以下全部在寄存器中完成 ----

    # 1. 均值
    mean = tl.sum(x, axis=0) / D

    # 2. 方差
    xmean = x - mean
    var = tl.sum(xmean * xmean, axis=0) / D

    # 3. 归一化
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = xmean * rstd

    # 4. Scale + Bias (affine transform)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = x_hat * w + b

    # 写回结果
    y_ptr = Y + row * stride
    tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=mask)

    # 保存中间量（backward 需要）
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


# ============================================================
# Backward Kernel
# ============================================================

@triton.jit
def layernorm_bwd_kernel(
    DY,       # 上游梯度 [N, D]
    X,        # 输入 [N, D]
    W,        # scale 参数 [D]
    Mean,     # forward 保存的均值 [N]
    Rstd,     # forward 保存的 1/std [N]
    DX,       # 输入梯度 [N, D]
    DW,       # scale 梯度 [D]（需要 atomic add）
    DB,       # bias 梯度 [D]（需要 atomic add）
    stride,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm backward 公式:
    dx_hat = dy * w
    dx = (1/D) * rstd * (D * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
    dw += dy * x_hat  (across all rows)
    db += dy          (across all rows)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # 加载数据
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # 计算 x_hat
    x_hat = (x - mean) * rstd

    # dx_hat = dy * w
    dx_hat = dy * w

    # dx = rstd * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
    c1 = tl.sum(dx_hat, axis=0) / D
    c2 = tl.sum(dx_hat * x_hat, axis=0) / D
    dx = rstd * (dx_hat - c1 - x_hat * c2)

    # 写回 dx
    tl.store(DX + row * stride + cols, dx.to(DX.dtype.element_ty), mask=mask)

    # dw, db 需要跨行累加（使用 atomic add）
    tl.atomic_add(DW + cols, (dy * x_hat).to(DW.dtype.element_ty), mask=mask)
    tl.atomic_add(DB + cols, dy.to(DB.dtype.element_ty), mask=mask)


# ============================================================
# Python wrapper
# ============================================================

class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        assert x.is_cuda
        N, D = x.shape
        BLOCK_SIZE = triton.next_power_of_2(D)

        y = torch.empty_like(x)
        mean = torch.empty(N, device=x.device, dtype=torch.float32)
        rstd = torch.empty(N, device=x.device, dtype=torch.float32)

        layernorm_fwd_kernel[(N,)](
            x, y, weight, bias, mean, rstd,
            x.stride(0), N, D, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.D = D
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        N, D = x.shape

        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        db = torch.zeros_like(weight, dtype=torch.float32)

        layernorm_bwd_kernel[(N,)](
            dy, x, weight, mean, rstd,
            dx, dw, db,
            x.stride(0), N, D,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        return dx, dw.to(weight.dtype), db.to(weight.dtype), None


def triton_layernorm(x, weight, bias, eps=1e-5):
    """Triton LayerNorm (支持 autograd)"""
    return TritonLayerNorm.apply(x, weight, bias, eps)


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    torch.manual_seed(0)
    configs = [
        (1024, 768),     # BERT-base hidden size
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),    # 大模型 hidden size
        (8192, 4096),
    ]

    print(f"{'Shape':>16} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)

    for N, D in configs:
        x = torch.randn(N, D, device='cuda', dtype=torch.float32)
        w = torch.randn(D, device='cuda', dtype=torch.float32)
        b = torch.randn(D, device='cuda', dtype=torch.float32)

        ln = torch.nn.LayerNorm(D, device='cuda', dtype=torch.float32)

        # 正确性验证
        ref = torch.nn.functional.layer_norm(x, (D,), w, b)
        out = triton_layernorm(x, w, b)
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-3), \
            f"Forward 不一致! max diff={torch.max(torch.abs(ref - out)).item()}"

        # 性能测试
        ms_torch = triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(x, (D,), w, b)
        )
        ms_triton = triton.testing.do_bench(
            lambda: triton_layernorm(x, w, b)
        )

        speedup = ms_torch / ms_triton
        shape_str = f"{N}x{D}"
        print(f"{shape_str:>16} | {ms_torch:>12.4f} | {ms_triton:>12.4f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: Layer Normalization")
    print("=" * 60)

    # Forward 测试
    print("\n--- Forward 功能测试 ---")
    N, D = 128, 768
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)
    w = torch.ones(D, device='cuda', dtype=torch.float32)
    b = torch.zeros(D, device='cuda', dtype=torch.float32)

    out = triton_layernorm(x, w, b)
    ref = torch.nn.functional.layer_norm(x, (D,), w, b)
    print(f"Input: [{N}, {D}]")
    print(f"Max diff vs PyTorch: {torch.max(torch.abs(ref - out)).item():.2e}")
    print(f"Output mean (should be ~0): {out.mean().item():.4f}")
    print(f"Output std  (should be ~1): {out.std().item():.4f}")

    # Backward 测试
    print("\n--- Backward 功能测试 ---")
    x = torch.randn(64, 256, device='cuda', dtype=torch.float32, requires_grad=True)
    w = torch.randn(256, device='cuda', dtype=torch.float32, requires_grad=True)
    b = torch.randn(256, device='cuda', dtype=torch.float32, requires_grad=True)
    dy = torch.randn(64, 256, device='cuda', dtype=torch.float32)

    # PyTorch reference
    ref_out = torch.nn.functional.layer_norm(x, (256,), w, b)
    ref_out.backward(dy)
    ref_dx, ref_dw, ref_db = x.grad.clone(), w.grad.clone(), b.grad.clone()

    # Triton
    x.grad, w.grad, b.grad = None, None, None
    tri_out = triton_layernorm(x, w, b)
    tri_out.backward(dy)

    print(f"dx max diff: {torch.max(torch.abs(ref_dx - x.grad)).item():.2e}")
    print(f"dw max diff: {torch.max(torch.abs(ref_dw - w.grad)).item():.2e}")
    print(f"db max diff: {torch.max(torch.abs(ref_db - b.grad)).item():.2e}")

    # Benchmark
    print("\n--- 性能对比 (Forward) ---")
    benchmark()
