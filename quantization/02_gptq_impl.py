"""
02_gptq_impl.py — 简化版 GPTQ 算法实现
=======================================

GPTQ (Generative Pre-trained Transformer Quantization) 的核心思想:
  逐列量化权重，利用 Hessian 信息将量化误差最优地补偿到未量化的列。

算法来源: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
          Generative Pre-trained Transformers", ICLR 2023

本文件实现一个简化版本，帮助理解 GPTQ 的核心逻辑:
  1. 用校准数据计算 Hessian (H = 2 * X^T * X)
  2. 对 Hessian 做 Cholesky 分解
  3. 逐列量化 + 误差补偿

运行: python 02_gptq_impl.py (不需要 GPU)
"""

import torch
import torch.nn as nn
import math
import time


# ================================================================
# 量化工具函数
# ================================================================

def quantize_value(w: float, scale: float, zero_point: int, qmin: int, qmax: int) -> int:
    """量化单个值"""
    return int(max(qmin, min(qmax, round(w / scale) + zero_point)))


def symmetric_quantize_tensor(tensor: torch.Tensor, bits: int = 4, group_size: int = -1):
    """
    对称量化 (支持 per-channel 和 per-group)

    Args:
        tensor: [out_features, in_features]
        bits: 量化位宽
        group_size: -1 = per-channel, >0 = per-group
    Returns:
        q_tensor, scale
    """
    qmax = 2 ** (bits - 1) - 1

    if group_size <= 0:
        # Per-channel
        abs_max = tensor.abs().amax(dim=1, keepdim=True)  # [out, 1]
        scale = abs_max / qmax
        scale = torch.clamp(scale, min=1e-8)
        q = torch.round(tensor / scale).clamp(-qmax - 1, qmax)
        return q, scale.squeeze(1)
    else:
        # Per-group
        out_features, in_features = tensor.shape
        assert in_features % group_size == 0
        num_groups = in_features // group_size

        w = tensor.reshape(out_features, num_groups, group_size)
        abs_max = w.abs().amax(dim=2, keepdim=True)
        scale = abs_max / qmax
        scale = torch.clamp(scale, min=1e-8)
        q = torch.round(w / scale).clamp(-qmax - 1, qmax)

        return q.reshape(out_features, in_features), scale.squeeze(2)


def dequantize_tensor(q_tensor, scale, group_size=-1):
    """反量化"""
    if group_size <= 0:
        return q_tensor.float() * scale[:, None]
    else:
        out_features, in_features = q_tensor.shape
        num_groups = in_features // group_size
        q = q_tensor.reshape(out_features, num_groups, group_size).float()
        w = q * scale[:, :, None]
        return w.reshape(out_features, in_features)


# ================================================================
# GPTQ 核心算法
# ================================================================

class GPTQ:
    """
    简化版 GPTQ 量化器

    核心流程:
    1. 收集校准数据，计算 Hessian H = 2 * X^T * X
    2. 对 H 做 Cholesky 分解: H = L * L^T
    3. 按列从左到右处理:
       a. 量化第 i 列
       b. 计算量化误差
       c. 用 Hessian 信息将误差最优分配到未量化列
    """

    def __init__(self, layer: nn.Linear, bits: int = 4, group_size: int = 128):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.out_features = layer.out_features
        self.in_features = layer.in_features

        # Hessian 累加器
        self.H = torch.zeros(self.in_features, self.in_features, dtype=torch.float32)
        self.n_samples = 0

    def add_calibration_data(self, inp: torch.Tensor):
        """
        添加校准数据到 Hessian

        inp: [batch, seq_len, in_features] 或 [batch, in_features]
        Hessian: H = 2 * X^T * X (X 是所有 token 的激活拼接)
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])  # [B*S, in]

        n = inp.shape[0]
        # H += X^T @ X (增量更新)
        self.H += inp.T @ inp
        self.n_samples += n

    def quantize(self, block_size: int = 128, percdamp: float = 0.01):
        """
        执行 GPTQ 量化

        Args:
            block_size: 每次处理的列数 (GPTQ 优化: 按 block 而非逐列处理)
            percdamp: Hessian 对角线阻尼 (防止数值不稳定)

        Returns:
            q_weight: 量化后的权重
            scale: 量化 scale
        """
        W = self.layer.weight.data.clone().float()  # [out, in]
        H = self.H / self.n_samples  # 归一化 Hessian

        # 添加阻尼: H += percdamp * diag(H) * I
        # 防止 Cholesky 分解失败
        damp = percdamp * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # Cholesky 分解: H = L * L^T
        # 我们需要 H^{-1}，但 GPTQ 只需要 Cholesky 因子
        try:
            L = torch.linalg.cholesky(H)
        except RuntimeError:
            print("  Cholesky 失败，增大阻尼重试...")
            H.diagonal().add_(damp * 10)
            L = torch.linalg.cholesky(H)

        # H 的逆的 Cholesky 因子
        Hinv = torch.cholesky_inverse(L)

        # 量化参数
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -(2 ** (self.bits - 1))

        # 存储量化结果
        Q = torch.zeros_like(W)

        # 按列处理的 scale
        if self.group_size > 0:
            num_groups = self.in_features // self.group_size
            scales = torch.zeros(self.out_features, num_groups)
        else:
            scales = torch.zeros(self.out_features)

        # GPTQ 核心: 逐 block 处理
        n_blocks = math.ceil(self.in_features / block_size)

        for block_idx in range(n_blocks):
            col_start = block_idx * block_size
            col_end = min(col_start + block_size, self.in_features)
            count = col_end - col_start

            # 当前 block 的权重和 Hessian 子块
            W_block = W[:, col_start:col_end].clone()
            Hinv_block = Hinv[col_start:col_end, col_start:col_end]
            Err = torch.zeros_like(W_block)

            # 逐列量化
            for j in range(count):
                global_col = col_start + j
                w_col = W_block[:, j]  # 当前列的所有行

                # 确定 scale (per-group 时根据当前列属于哪个 group)
                if self.group_size > 0 and global_col % self.group_size == 0:
                    # 计算这个 group 的 scale
                    g = global_col // self.group_size
                    g_end = min(global_col + self.group_size, self.in_features)
                    group_w = W[:, global_col:g_end]
                    abs_max = group_w.abs().amax(dim=1)
                    cur_scale = abs_max / qmax
                    cur_scale = torch.clamp(cur_scale, min=1e-8)
                    scales[:, g] = cur_scale

                if self.group_size > 0:
                    g = global_col // self.group_size
                    cur_scale = scales[:, g]
                else:
                    cur_scale = w_col.abs().max() / qmax
                    cur_scale = max(cur_scale.item(), 1e-8)
                    if j == 0:
                        # Per-channel: 在第一列时计算整行的 scale
                        abs_max = W.abs().amax(dim=1)
                        per_ch_scale = abs_max / qmax
                        per_ch_scale = torch.clamp(per_ch_scale, min=1e-8)
                        scales = per_ch_scale
                    cur_scale = scales

                # 量化当前列
                if isinstance(cur_scale, torch.Tensor):
                    q_col = torch.round(w_col / cur_scale).clamp(qmin, qmax)
                    dq_col = q_col * cur_scale  # dequantized
                else:
                    q_col = torch.round(w_col / cur_scale).clamp(qmin, qmax)
                    dq_col = q_col * cur_scale

                Q[:, global_col] = q_col

                # 计算量化误差
                err = (w_col - dq_col) / Hinv_block[j, j]

                # 误差补偿: 把误差分配到 block 内未量化的列
                if j + 1 < count:
                    W_block[:, j + 1:] -= err[:, None] @ Hinv_block[j, j + 1:].unsqueeze(0)

                Err[:, j] = err

            # 把误差补偿到后续 block
            if col_end < self.in_features:
                W[:, col_end:] -= Err @ Hinv[col_start:col_end, col_end:]

        return Q.to(torch.int8), scales


# ================================================================
# RTN 基线 (用于对比)
# ================================================================

def rtn_quantize(weight: torch.Tensor, bits: int = 4, group_size: int = 128):
    """
    Round-To-Nearest 量化 (最简单的基线)
    直接四舍五入，不做误差补偿
    """
    return symmetric_quantize_tensor(weight, bits=bits, group_size=group_size)


# ================================================================
# 对比实验
# ================================================================

def compare_gptq_vs_rtn():
    """
    对比 GPTQ 和 RTN 的精度差异
    模拟一个 MLP 层的量化
    """
    torch.manual_seed(42)

    # 模拟一个中等大小的 Linear 层
    in_features = 1024
    out_features = 1024
    batch_size = 32
    seq_len = 64

    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(layer.weight, std=0.02)

    # 给一些 channel 添加 outlier（模拟真实模型）
    with torch.no_grad():
        layer.weight[0] *= 5
        layer.weight[10] *= 3

    # 生成校准数据
    calibration_data = torch.randn(batch_size, seq_len, in_features)

    # 参考输出 (FP32)
    test_input = torch.randn(8, 16, in_features)
    with torch.no_grad():
        ref_output = layer(test_input)

    print("=" * 85)
    print("GPTQ vs RTN 精度对比")
    print(f"Layer: Linear({in_features}, {out_features})")
    print("=" * 85)

    for bits in [4, 8]:
        for group_size in [128, 64]:
            print(f"\n--- {bits}-bit, group_size={group_size} ---")

            # RTN 量化
            q_rtn, scale_rtn = rtn_quantize(layer.weight.data, bits=bits, group_size=group_size)
            w_rtn = dequantize_tensor(q_rtn, scale_rtn, group_size=group_size)
            with torch.no_grad():
                out_rtn = test_input @ w_rtn.T

            mse_rtn = ((ref_output - out_rtn) ** 2).mean().item()
            cos_rtn = torch.cosine_similarity(
                ref_output.flatten(), out_rtn.flatten(), dim=0
            ).item()

            # GPTQ 量化
            gptq = GPTQ(layer, bits=bits, group_size=group_size)
            gptq.add_calibration_data(calibration_data)

            t0 = time.time()
            q_gptq, scale_gptq = gptq.quantize(block_size=128)
            t_gptq = time.time() - t0

            w_gptq = dequantize_tensor(q_gptq, scale_gptq, group_size=group_size)
            with torch.no_grad():
                out_gptq = test_input @ w_gptq.T

            mse_gptq = ((ref_output - out_gptq) ** 2).mean().item()
            cos_gptq = torch.cosine_similarity(
                ref_output.flatten(), out_gptq.flatten(), dim=0
            ).item()

            print(f"  {'方法':<12} | {'Output MSE':>12} | {'CosineSim':>10} | {'量化耗时':>10}")
            print(f"  {'-'*52}")
            print(f"  {'RTN':<12} | {mse_rtn:>12.2e} | {cos_rtn:>10.6f} | {'< 1ms':>10}")
            print(f"  {'GPTQ':<12} | {mse_gptq:>12.2e} | {cos_gptq:>10.6f} | {t_gptq:>9.3f}s")

            improvement = mse_rtn / (mse_gptq + 1e-15)
            print(f"  GPTQ MSE 降低: {improvement:.1f}x")

    print("\n--- 结论 ---")
    print("1. INT4 下 GPTQ 比 RTN 精度显著提升（通常 MSE 降低 2-10x）")
    print("2. INT8 下差距较小（因为 INT8 的量化级别够多，RTN 误差本来就小）")
    print("3. GPTQ 的代价是量化速度慢（需要计算 Hessian + 逐列处理）")
    print("4. 这就是为什么 GPTQ 主要用于 INT4/INT3 这种低比特量化")


def explain_gptq_intuition():
    """
    用一个极简示例解释 GPTQ 的直觉
    """
    print("\n" + "=" * 85)
    print("GPTQ 直觉解释 (极简 2x2 示例)")
    print("=" * 85)

    # 一个 2x2 的 "权重矩阵"
    W = torch.tensor([[1.3, 0.7],
                      [0.4, 2.1]])

    print(f"原始权重:\n{W}")
    print(f"\n假设 INT4 对称量化 (qmax=7):")

    # RTN: 直接量化
    qmax = 7
    scale = W.abs().amax(dim=1, keepdim=True) / qmax
    W_rtn = torch.round(W / scale) * scale
    print(f"\n--- RTN (直接四舍五入) ---")
    print(f"量化后: \n{W_rtn}")
    print(f"误差:   \n{(W - W_rtn).abs()}")
    print(f"总 MSE: {((W - W_rtn) ** 2).mean():.6f}")

    # GPTQ: 量化第一列时，把误差补偿到第二列
    print(f"\n--- GPTQ (误差补偿) ---")
    W_gptq = W.clone()

    # 假设 Hessian 对角线 = [1.0, 1.0] (简化)
    # 量化第 0 列
    q0 = torch.round(W_gptq[:, 0] / scale.squeeze())
    q0 = q0.clamp(-qmax - 1, qmax)
    dq0 = q0 * scale.squeeze()
    err0 = W_gptq[:, 0] - dq0
    print(f"第 0 列: 量化误差 = {err0.tolist()}")

    # 误差补偿到第 1 列
    W_gptq[:, 1] += err0  # 简化: 假设 H[0,1]/H[0,0] = 1
    print(f"第 1 列 (补偿后): {W_gptq[:, 1].tolist()}")

    # 量化第 1 列
    q1 = torch.round(W_gptq[:, 1] / scale.squeeze())
    q1 = q1.clamp(-qmax - 1, qmax)
    dq1 = q1 * scale.squeeze()

    W_result = torch.stack([dq0, dq1], dim=1)
    print(f"量化后: \n{W_result}")
    print(f"误差:   \n{(W - W_result).abs()}")
    print(f"总 MSE: {((W - W_result) ** 2).mean():.6f}")
    print(f"\n关键洞察: GPTQ 把第一列的误差'转嫁'给了第二列")
    print("Hessian 告诉我们: 哪些列对输出影响大，误差应该优先从那些列转走")


# ================================================================
# 主函数
# ================================================================

if __name__ == "__main__":
    print("=" * 85)
    print("02 GPTQ Implementation: 理解 GPTQ 量化算法")
    print("=" * 85)

    # 1. 直觉解释
    explain_gptq_intuition()

    # 2. 精度对比
    compare_gptq_vs_rtn()
