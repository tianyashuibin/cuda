"""
Step 4: Flash Attention
========================
Triton 的杀手级应用：用 ~120 行 Python 实现 Flash Attention
（原版 CUDA 实现 ~2000 行）

Flash Attention 核心思想：
  1. 不存储完整的 N×N attention 矩阵（O(N²) 内存 → O(N) 内存）
  2. 分块遍历 K/V，在线更新 softmax 的统计量 (max, sum)
  3. 所有操作 fused 在一个 kernel 里：QK^T → scale → mask → softmax → @V

为什么 Triton 适合：
  - Flash Attention 的关键在于 kernel fusion
  - 手写 CUDA 需要精心管理 shared memory layout、warp-level reduction 等
  - Triton 只需要写算法逻辑，编译器处理底层细节

实现说明：
  这里实现的是简化版 Flash Attention (forward only, causal mask)
  完整版还包括 backward pass 和更多优化，可参考 Triton 官方教程
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q 的 strides: [batch, head, seq, dim]
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    N_CTX,         # 序列长度
    BLOCK_M: tl.constexpr,   # Q 的 tile 大小
    BLOCK_N: tl.constexpr,   # K/V 的 tile 大小
    BLOCK_DMODEL: tl.constexpr,  # head dimension (必须是 2 的幂)
    IS_CAUSAL: tl.constexpr,
):
    # 每个 program 处理一个 (batch, head, Q_block) 的组合
    start_m = tl.program_id(0)  # Q 的 block 编号
    off_hz = tl.program_id(1)   # batch * n_heads 的线性索引

    # 计算 Q/K/V/Out 的基地址偏移
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    # ---- 加载 Q 的一个 tile [BLOCK_M, BLOCK_DMODEL] ----
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # scale factor: 1/sqrt(d_k)
    sm_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_DMODEL, tl.float32))

    # ---- Online Softmax 统计量 ----
    # m_i: 每行的 running max (用于数值稳定)
    # l_i: 每行的 running sum of exp
    # acc: 加权累加 softmax(QK^T) @ V
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # ---- 分块遍历 K/V（Flash Attention 的核心循环）----
    # 对比标准 attention: 需要先算完整个 N×N 的 QK^T 再做 softmax
    # Flash Attention: 每次只算 BLOCK_N 列，在线更新 softmax
    if IS_CAUSAL:
        # Causal mask: Q[i] 只能 attend 到 K[0:i+1]
        # 所以 Q block start_m 最远只需要看到 K block (start_m+1)*BLOCK_M
        n_blocks = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    else:
        n_blocks = tl.cdiv(N_CTX, BLOCK_N)

    for block_n in range(0, n_blocks):
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # 加载 K tile [BLOCK_N, BLOCK_DMODEL]
        k_ptrs = K + k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # QK^T: [BLOCK_M, BLOCK_N] = Q[BLOCK_M, D] @ K^T[D, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Causal mask: 把 Q[i] 不该看到的 K[j>i] 位置设为 -inf
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # ---- Online Softmax Update ----
        # 标准 softmax: exp(x - max) / sum(exp(x - max))
        # Online 版本: 当新的 block 来了一个更大的 max 时，需要修正之前的累加值
        #
        # 算法：
        #   m_new = max(m_old, max(qk))
        #   correction = exp(m_old - m_new)    ← 之前累加值的修正系数
        #   l_new = correction * l_old + sum(exp(qk - m_new))
        #   acc_new = correction * acc_old + exp(qk - m_new) @ V

        m_ij = tl.max(qk, axis=1)                    # 当前 block 的行最大值
        m_new = tl.maximum(m_i, m_ij)                 # 全局 running max
        alpha = tl.exp(m_i - m_new)                   # 修正系数
        p = tl.exp(qk - m_new[:, None])               # softmax 分子

        # 更新统计量
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc  # 修正之前的累加值

        # 加载 V tile [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # 累加 softmax(QK^T) @ V
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ---- 最终归一化 ----
    acc = acc / l_i[:, None]

    # ---- 写回结果 ----
    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention(q, k, v, causal=True):
    """
    Triton Flash Attention

    Args:
        q, k, v: [batch, n_heads, seq_len, head_dim], dtype=float16
        causal: 是否使用 causal mask
    Returns:
        output: [batch, n_heads, seq_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    B, H, N, D = q.shape
    assert D in {16, 32, 64, 128}, f"head_dim must be 16/32/64/128, got {D}"

    output = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N, BLOCK_M), B * H)

    flash_attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
    )
    return output


def reference_attention(q, k, v, causal=True):
    """标准 attention 实现（用于正确性验证）"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        N = q.shape[2]
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v.float()).to(q.dtype)


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    """对比 Flash Attention vs PyTorch SDPA"""
    torch.manual_seed(0)
    B, H, D = 4, 32, 64
    seq_lengths = [256, 512, 1024, 2048, 4096]

    print(f"{'SeqLen':>8} | {'PyTorch SDPA (ms)':>17} | {'Triton FA (ms)':>14} | {'Speedup':>8}")
    print("-" * 60)

    for N in seq_lengths:
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

        # 正确性验证（用小矩阵）
        if N <= 1024:
            ref = reference_attention(q, k, v, causal=True)
            out = flash_attention(q, k, v, causal=True)
            max_diff = torch.max(torch.abs(ref.float() - out.float())).item()
            assert max_diff < 1e-2, f"结果不一致! max diff={max_diff}"

        # 性能测试
        ms_sdpa = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        )
        ms_triton = triton.testing.do_bench(
            lambda: flash_attention(q, k, v, causal=True)
        )

        speedup = ms_sdpa / ms_triton
        print(f"{N:>8} | {ms_sdpa:>17.4f} | {ms_triton:>14.4f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Flash Attention")
    print("=" * 60)

    # 基本功能测试
    print("\n--- 功能测试 ---")
    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out = flash_attention(q, k, v, causal=True)
    ref = reference_attention(q, k, v, causal=True)
    print(f"Shape: Q/K/V = [{B}, {H}, {N}, {D}]")
    print(f"Output shape: {out.shape}")
    print(f"Max diff vs reference: {torch.max(torch.abs(ref.float() - out.float())).item():.2e}")

    # Benchmark
    print("\n--- 性能对比 (B=4, H=32, D=64, causal=True) ---")
    benchmark()
