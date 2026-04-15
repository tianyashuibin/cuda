"""
01_ptq_basics.py — 从零实现 Post-Training Quantization (PTQ)
=============================================================

学习目标:
  1. 理解对称 vs 非对称量化的区别
  2. 理解 per-tensor / per-channel / per-group 的区别
  3. 动手实现量化 + 反量化
  4. 对比不同量化配置对精度的影响

运行: python 01_ptq_basics.py (不需要 GPU，纯 CPU 即可)
"""

import torch
import torch.nn as nn
import math


# ================================================================
# 第一部分: 量化基础函数
# ================================================================

def symmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple:
    """
    对称量化: x_q = round(x / scale), scale = max(|x|) / qmax

    Args:
        tensor: 输入 float tensor
        bits: 量化位宽
    Returns:
        (quantized_tensor, scale)
    """
    qmax = 2 ** (bits - 1) - 1   # INT8: 127, INT4: 7
    qmin = -(2 ** (bits - 1))     # INT8: -128, INT4: -8

    # 计算 scale
    abs_max = tensor.abs().max()
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)  # 防止除零

    # 量化
    q = torch.round(tensor / scale)
    q = torch.clamp(q, qmin, qmax)

    return q.to(torch.int8 if bits == 8 else torch.int32), scale


def asymmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple:
    """
    非对称量化: x_q = round(x / scale) + zero_point

    非对称量化能更好地利用量化范围，适合分布不对称的激活值
    """
    qmax = 2 ** bits - 1   # UINT8: 255
    qmin = 0

    # 计算 scale 和 zero_point
    xmin = tensor.min()
    xmax = tensor.max()
    scale = (xmax - xmin) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = torch.round(-xmin / scale).to(torch.int32)

    # 量化
    q = torch.round(tensor / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)

    return q.to(torch.uint8 if bits == 8 else torch.int32), scale, zero_point


def symmetric_dequantize(q_tensor, scale):
    """反量化: x = x_q * scale"""
    return q_tensor.float() * scale


def asymmetric_dequantize(q_tensor, scale, zero_point):
    """反量化: x = (x_q - zero_point) * scale"""
    return (q_tensor.float() - zero_point.float()) * scale


# ================================================================
# 第二部分: 不同粒度的量化
# ================================================================

def quantize_per_tensor(weight: torch.Tensor, bits: int = 8):
    """
    Per-Tensor 量化: 整个 tensor 共享一个 scale
    最简单，但精度最差 — 不同 channel 的值域差异被忽略
    """
    qmax = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / qmax
    scale = max(scale.item(), 1e-8)

    q = torch.round(weight / scale).clamp(-qmax - 1, qmax)
    return q.to(torch.int8), torch.tensor(scale)


def quantize_per_channel(weight: torch.Tensor, bits: int = 8, axis: int = 0):
    """
    Per-Channel 量化: 每个 output channel 一个 scale
    最常用的权重量化方式

    weight: [out_features, in_features]
    axis=0: 每行一个 scale (per output channel)
    """
    qmax = 2 ** (bits - 1) - 1

    # 沿指定轴计算 abs max
    abs_max = weight.abs().amax(dim=1 if axis == 0 else 0)  # [out_features]
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)

    # 量化
    if axis == 0:
        q = torch.round(weight / scale[:, None])
    else:
        q = torch.round(weight / scale[None, :])
    q = q.clamp(-qmax - 1, qmax)

    return q.to(torch.int8), scale


def quantize_per_group(weight: torch.Tensor, bits: int = 4, group_size: int = 128):
    """
    Per-Group 量化: 每 group_size 个元素共享一个 scale
    INT4 量化必备 — 只有 16 个量化级别，per-channel 粒度不够

    weight: [out_features, in_features]
    group_size: 每组元素数 (常用 32, 64, 128)
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, f"in_features ({in_features}) 必须整除 group_size ({group_size})"

    qmax = 2 ** (bits - 1) - 1  # INT4: 7
    num_groups = in_features // group_size

    # reshape: [out, in] -> [out, num_groups, group_size]
    w = weight.reshape(out_features, num_groups, group_size)

    # 每组一个 scale
    abs_max = w.abs().amax(dim=2)  # [out, num_groups]
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)

    # 量化
    q = torch.round(w / scale[:, :, None])
    q = q.clamp(-qmax - 1, qmax)

    # reshape 回去
    q = q.reshape(out_features, in_features)

    return q.to(torch.int8), scale  # scale shape: [out, num_groups]


def dequantize_per_group(q_weight, scale, group_size):
    """Per-Group 反量化"""
    out_features, in_features = q_weight.shape
    num_groups = in_features // group_size

    q = q_weight.reshape(out_features, num_groups, group_size).float()
    w = q * scale[:, :, None]
    return w.reshape(out_features, in_features)


# ================================================================
# 第三部分: 精度对比实验
# ================================================================

def compute_quantization_error(original, dequantized):
    """计算量化误差"""
    mse = ((original - dequantized) ** 2).mean().item()
    snr = 10 * math.log10(original.var().item() / (mse + 1e-10))
    max_err = (original - dequantized).abs().max().item()
    return mse, snr, max_err


def compare_quantization_methods():
    """对比不同量化方式的精度"""
    torch.manual_seed(42)

    # 模拟一个 Linear 层的权重 (类似 LLaMA-7B 的 FFN)
    out_features, in_features = 4096, 4096
    # 真实权重分布通常接近正态分布
    weight = torch.randn(out_features, in_features) * 0.02

    # 模拟一些 outlier channels (真实模型中常见)
    weight[0] *= 10    # 第 0 行的值特别大
    weight[1] *= 8
    weight[2] *= 5

    print("=" * 85)
    print("量化精度对比实验")
    print(f"权重 shape: [{out_features}, {in_features}]")
    print(f"权重范围: [{weight.min():.4f}, {weight.max():.4f}]")
    print("=" * 85)

    results = []

    # --- INT8 量化 ---
    print("\n--- INT8 (8-bit) 量化 ---")
    print(f"{'方法':<30} | {'MSE':>12} | {'SNR (dB)':>10} | {'Max Error':>12}")
    print("-" * 72)

    # Per-Tensor INT8
    q, scale = quantize_per_tensor(weight, bits=8)
    deq = q.float() * scale
    mse, snr, max_err = compute_quantization_error(weight, deq)
    print(f"{'Per-Tensor INT8':<30} | {mse:>12.2e} | {snr:>10.2f} | {max_err:>12.2e}")
    results.append(("Per-Tensor INT8", mse, snr))

    # Per-Channel INT8
    q, scale = quantize_per_channel(weight, bits=8, axis=0)
    deq = q.float() * scale[:, None]
    mse, snr, max_err = compute_quantization_error(weight, deq)
    print(f"{'Per-Channel INT8':<30} | {mse:>12.2e} | {snr:>10.2f} | {max_err:>12.2e}")
    results.append(("Per-Channel INT8", mse, snr))

    # --- INT4 量化 ---
    print("\n--- INT4 (4-bit) 量化 ---")
    print(f"{'方法':<30} | {'MSE':>12} | {'SNR (dB)':>10} | {'Max Error':>12}")
    print("-" * 72)

    # Per-Tensor INT4
    q_t, scale_t = quantize_per_tensor(weight, bits=4)
    deq = q_t.float() * scale_t
    mse, snr, max_err = compute_quantization_error(weight, deq)
    print(f"{'Per-Tensor INT4':<30} | {mse:>12.2e} | {snr:>10.2f} | {max_err:>12.2e}")
    results.append(("Per-Tensor INT4", mse, snr))

    # Per-Channel INT4
    q_c, scale_c = quantize_per_channel(weight, bits=4, axis=0)
    deq = q_c.float() * scale_c[:, None]
    mse, snr, max_err = compute_quantization_error(weight, deq)
    print(f"{'Per-Channel INT4':<30} | {mse:>12.2e} | {snr:>10.2f} | {max_err:>12.2e}")
    results.append(("Per-Channel INT4", mse, snr))

    # Per-Group INT4 (不同 group_size)
    for gs in [128, 64, 32]:
        q_g, scale_g = quantize_per_group(weight, bits=4, group_size=gs)
        deq = dequantize_per_group(q_g, scale_g, gs)
        mse, snr, max_err = compute_quantization_error(weight, deq)
        label = f"Per-Group INT4 (gs={gs})"
        print(f"{label:<30} | {mse:>12.2e} | {snr:>10.2f} | {max_err:>12.2e}")
        results.append((label, mse, snr))

    # --- 结论 ---
    print("\n--- 结论 ---")
    print("1. Per-Tensor → Per-Channel → Per-Group，精度逐步提升")
    print("2. INT4 Per-Tensor 基本不可用（SNR 太低），必须 Per-Group")
    print("3. group_size 越小精度越好，但 scale 的存储开销越大")
    print("4. group_size=128 是 GPTQ/AWQ 的默认值（精度与开销的平衡点）")
    print("5. 有 outlier channel 时，Per-Channel/Per-Group 的优势更明显")


def simulate_linear_quantization():
    """
    模拟一个完整 Linear 层的量化推理过程
    展示量化如何影响最终输出
    """
    torch.manual_seed(42)

    print("\n" + "=" * 85)
    print("Linear 层量化推理模拟")
    print("=" * 85)

    # 模拟 LLaMA-7B 的一个 FFN 层
    in_features = 4096
    out_features = 11008  # LLaMA up/gate projection
    batch_size = 32
    seq_len = 1  # decode 场景

    # 原始权重和输入
    weight = torch.randn(out_features, in_features) * 0.02
    x = torch.randn(batch_size, seq_len, in_features) * 0.1
    x_2d = x.reshape(-1, in_features)  # [B*S, in]

    # 参考输出 (FP32)
    ref_output = x_2d @ weight.T  # [B*S, out]

    print(f"输入:  [{batch_size}x{seq_len}, {in_features}]")
    print(f"权重:  [{out_features}, {in_features}]")
    print(f"输出:  [{batch_size}x{seq_len}, {out_features}]")

    print(f"\n{'量化方式':<30} | {'输出 MSE':>12} | {'输出 CosSim':>12} | {'权重大小 (MB)':>14}")
    print("-" * 76)

    # FP16 (基线)
    w_fp16 = weight.half()
    out_fp16 = x_2d @ w_fp16.float().T
    mse = ((ref_output - out_fp16) ** 2).mean().item()
    cos = torch.cosine_similarity(ref_output.flatten(), out_fp16.flatten(), dim=0).item()
    size_mb = out_features * in_features * 2 / 1024 / 1024
    print(f"{'FP16 (无量化)':<30} | {mse:>12.2e} | {cos:>12.6f} | {size_mb:>14.1f}")

    # INT8 Per-Channel
    q, scale = quantize_per_channel(weight, bits=8, axis=0)
    w_deq = q.float() * scale[:, None]
    out_q = x_2d @ w_deq.T
    mse = ((ref_output - out_q) ** 2).mean().item()
    cos = torch.cosine_similarity(ref_output.flatten(), out_q.flatten(), dim=0).item()
    size_mb = (out_features * in_features * 1 + out_features * 4) / 1024 / 1024  # int8 + scale
    print(f"{'INT8 Per-Channel':<30} | {mse:>12.2e} | {cos:>12.6f} | {size_mb:>14.1f}")

    # INT4 Per-Group (gs=128)
    for gs in [128, 64]:
        q, scale = quantize_per_group(weight, bits=4, group_size=gs)
        w_deq = dequantize_per_group(q, scale, gs)
        out_q = x_2d @ w_deq.T
        mse = ((ref_output - out_q) ** 2).mean().item()
        cos = torch.cosine_similarity(ref_output.flatten(), out_q.flatten(), dim=0).item()
        num_groups = in_features // gs
        # INT4: 0.5 byte per weight + scale (fp16) per group
        size_mb = (out_features * in_features * 0.5 + out_features * num_groups * 2) / 1024 / 1024
        label = f"INT4 Per-Group (gs={gs})"
        print(f"{label:<30} | {mse:>12.2e} | {cos:>12.6f} | {size_mb:>14.1f}")

    print("\n--- 结论 ---")
    print("Cosine Similarity > 0.999 表示量化模型输出几乎与原始一致")
    print("INT4 Per-Group(128) 能在 ~4x 压缩下保持极高的输出相似度")
    print("这就是为什么 GPTQ/AWQ (INT4 + per-group) 是目前最主流的量化方案")


# ================================================================
# 第四部分: INT4 打包 (Packing) 演示
# ================================================================

def demo_int4_packing():
    """
    演示 INT4 权重的打包和解包
    这是 CUDA 量化 kernel 的基础
    """
    print("\n" + "=" * 85)
    print("INT4 打包 (Packing) 演示")
    print("=" * 85)

    # 8 个 INT4 值 (范围 -8 ~ 7)
    values = torch.tensor([3, -2, 7, -8, 0, 5, -1, 4], dtype=torch.int8)
    print(f"原始 INT4 值: {values.tolist()}")
    print(f"二进制表示:")
    for v in values:
        # 4-bit 无符号表示
        unsigned = v.item() & 0xF
        print(f"  {v.item():>3d} → 0b{unsigned:04b} (unsigned: {unsigned})")

    # 打包: 每两个 INT4 放入一个 INT8
    packed = torch.zeros(len(values) // 2, dtype=torch.uint8)
    for i in range(0, len(values), 2):
        low = values[i].item() & 0xF    # 低 4 bit
        high = values[i+1].item() & 0xF  # 高 4 bit
        packed[i // 2] = low | (high << 4)

    print(f"\n打包后 (4 个 uint8): {packed.tolist()}")
    print(f"存储节省: {len(values)} bytes → {len(packed)} bytes (2x 压缩)")

    # 解包
    unpacked = torch.zeros(len(values), dtype=torch.int8)
    for i in range(len(packed)):
        p = packed[i].item()
        low = p & 0xF
        high = (p >> 4) & 0xF
        # 符号扩展: 如果第 3 bit 是 1，说明是负数
        if low >= 8: low -= 16
        if high >= 8: high -= 16
        unpacked[2 * i] = low
        unpacked[2 * i + 1] = high

    print(f"解包后: {unpacked.tolist()}")
    print(f"验证: 解包结果与原始一致 = {torch.equal(values, unpacked)}")

    # 更高效的向量化打包 (实际 kernel 中的做法)
    print("\n--- 向量化打包 (kernel 中的做法) ---")
    # 用 int32 存 8 个 INT4
    vals = values.to(torch.int32) & 0xF
    packed_i32 = torch.tensor(0, dtype=torch.int32)
    for i in range(8):
        packed_i32 |= (vals[i] << (i * 4))
    print(f"8 个 INT4 打包到 1 个 int32: 0x{packed_i32.item():08X}")
    print(f"存储: 8 bytes → 4 bytes")


# ================================================================
# 主函数
# ================================================================

if __name__ == "__main__":
    print("=" * 85)
    print("01 PTQ Basics: 从零理解量化")
    print("=" * 85)

    # 1. 基础量化演示
    print("\n--- 基础量化演示 ---")
    x = torch.tensor([1.2, -0.5, 3.7, -2.1, 0.0, 1.8])
    print(f"原始 FP32: {x.tolist()}")

    # 对称量化
    q_sym, scale_sym = symmetric_quantize(x, bits=8)
    deq_sym = symmetric_dequantize(q_sym, scale_sym)
    print(f"对称 INT8:  {q_sym.tolist()}, scale={scale_sym.item():.6f}")
    print(f"反量化:     {deq_sym.tolist()}")
    print(f"误差:       {(x - deq_sym).abs().tolist()}")

    # 非对称量化
    q_asym, scale_asym, zp = asymmetric_quantize(x, bits=8)
    deq_asym = asymmetric_dequantize(q_asym, scale_asym, zp)
    print(f"\n非对称 UINT8: {q_asym.tolist()}, scale={scale_asym.item():.6f}, zp={zp.item()}")
    print(f"反量化:       {[f'{v:.4f}' for v in deq_asym.tolist()]}")

    # 2. 不同方式精度对比
    compare_quantization_methods()

    # 3. Linear 层量化模拟
    simulate_linear_quantization()

    # 4. INT4 打包演示
    demo_int4_packing()
