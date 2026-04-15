"""
03_awq_impl.py — 简化版 AWQ (Activation-Aware Quantization) 实现
================================================================

AWQ 的核心思想:
  不是所有权重通道同等重要。少数通道对应着大激活值，
  量化这些通道的误差对模型输出影响更大。
  AWQ 通过 per-channel scaling 保护这些 "salient channels"。

算法来源: Lin et al., "AWQ: Activation-aware Weight Quantization for
          LLM Compression and Acceleration", MLSys 2024

与 GPTQ 的关键区别:
  - GPTQ: 逐列处理，用 Hessian 做误差补偿 → 精度好，但量化慢
  - AWQ:  per-channel scaling，搜索最优 scale → 量化快，精度与 GPTQ 相当

运行: python 03_awq_impl.py (不需要 GPU)
"""

import torch
import torch.nn as nn
import math


# ================================================================
# 量化工具函数 (与 01/02 相同)
# ================================================================

def symmetric_quantize(tensor: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Per-Group 对称量化"""
    qmax = 2 ** (bits - 1) - 1
    out_features, in_features = tensor.shape

    if group_size <= 0 or group_size >= in_features:
        # Per-channel
        abs_max = tensor.abs().amax(dim=1, keepdim=True)
        scale = torch.clamp(abs_max / qmax, min=1e-8)
        q = torch.round(tensor / scale).clamp(-qmax - 1, qmax)
        return q, scale.squeeze(1)

    # Per-group
    num_groups = in_features // group_size
    w = tensor.reshape(out_features, num_groups, group_size)
    abs_max = w.abs().amax(dim=2, keepdim=True)
    scale = torch.clamp(abs_max / qmax, min=1e-8)
    q = torch.round(w / scale).clamp(-qmax - 1, qmax)
    return q.reshape(out_features, in_features), scale.squeeze(2)


def dequantize(q_tensor, scale, group_size=128):
    """反量化"""
    if scale.dim() == 1:
        return q_tensor.float() * scale[:, None]

    out_features, in_features = q_tensor.shape
    num_groups = in_features // group_size
    q = q_tensor.reshape(out_features, num_groups, group_size).float()
    w = q * scale[:, :, None]
    return w.reshape(out_features, in_features)


# ================================================================
# AWQ 核心算法
# ================================================================

class AWQ:
    """
    简化版 AWQ 量化器

    核心流程:
    1. 用校准数据统计每个通道的激活值大小
    2. 对权重做 per-channel scaling: w_scaled = w * s
       其中 s 对 salient channels 放大权重(使量化更精细)
    3. 搜索最优的 scaling factor α: s = activation_scale ^ α
    4. 量化 scaled weight
    5. 推理时: output = x @ (Q * scale / s)  (s 的逆可以融合进 scale)

    关键公式:
      原始: Y = X @ W
      AWQ:  Y = (X / s) @ (s * W)  → 量化 (s * W) 而非 W
      为什么: s * W 使得 salient channels 的值更大
              → 量化分辨率更高 → 误差更小
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.activation_scales = {}  # layer_name -> scale

    def compute_activation_scale(self, activations: torch.Tensor) -> torch.Tensor:
        """
        计算每个通道的激活值大小 (用于确定 salient channels)

        activations: [n_samples, seq_len, in_features] 或 [n_tokens, in_features]
        returns: [in_features] — 每个通道的平均绝对值
        """
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        # 使用平均绝对值作为 "重要性" 指标
        return activations.abs().mean(dim=0)  # [in_features]

    def search_best_alpha(
        self,
        weight: torch.Tensor,
        activation_scale: torch.Tensor,
        calibration_input: torch.Tensor,
        n_grid: int = 20,
    ) -> float:
        """
        搜索最优 α: s = activation_scale ^ α

        α 控制 scaling 的强度:
          α = 0: 不做 scaling (退化为 RTN)
          α = 1: 完全按激活大小 scaling
          最优 α 通常在 0.4 ~ 0.6 之间

        搜索目标: 最小化量化后的输出误差
        """
        if calibration_input.dim() == 3:
            calibration_input = calibration_input.reshape(-1, calibration_input.shape[-1])

        # FP32 参考输出
        ref_output = calibration_input @ weight.T  # [n_tokens, out]

        best_alpha = 0.0
        best_mse = float('inf')

        for i in range(n_grid + 1):
            alpha = i / n_grid  # 0.0 ~ 1.0

            # 计算 scaling factor
            s = activation_scale.pow(alpha)
            s = torch.clamp(s, min=1e-5)  # 防止极端值

            # 对权重做 scaling
            w_scaled = weight * s[None, :]  # [out, in] * [in] → broadcast

            # 量化 scaled weight
            q, scale = symmetric_quantize(w_scaled, bits=self.bits, group_size=self.group_size)
            w_deq = dequantize(q, scale, group_size=self.group_size)

            # 反向 scaling (推理时需要)
            w_deq = w_deq / s[None, :]

            # 计算输出误差
            out = calibration_input @ w_deq.T
            mse = ((ref_output - out) ** 2).mean().item()

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        return best_alpha

    def quantize_weight(
        self,
        weight: torch.Tensor,
        activation_scale: torch.Tensor,
        alpha: float,
    ):
        """
        用最优 α 量化权重

        Returns:
            q_weight: 量化后的权重 (int8 存储)
            q_scale: 量化 scale (包含了 activation scaling 的补偿)
            channel_scale: per-channel activation scale (用于推理时除回去)
        """
        # 计算 scaling factor
        s = activation_scale.pow(alpha)
        s = torch.clamp(s, min=1e-5)

        # Scale 权重
        w_scaled = weight * s[None, :]

        # 量化
        q, scale = symmetric_quantize(w_scaled, bits=self.bits, group_size=self.group_size)

        return q, scale, s

    def quantize_layer(
        self,
        layer: nn.Linear,
        calibration_data: torch.Tensor,
        n_grid: int = 20,
    ):
        """
        完整的 AWQ 量化流程 (针对单个 Linear 层)
        """
        weight = layer.weight.data.float()

        # Step 1: 计算激活 scale
        act_scale = self.compute_activation_scale(calibration_data)

        # Step 2: 搜索最优 α
        alpha = self.search_best_alpha(
            weight, act_scale, calibration_data, n_grid=n_grid
        )

        # Step 3: 用最优 α 量化
        q_weight, q_scale, channel_scale = self.quantize_weight(weight, act_scale, alpha)

        return q_weight, q_scale, channel_scale, alpha


# ================================================================
# 对比实验
# ================================================================

def compare_awq_vs_rtn():
    """对比 AWQ 和 RTN 的精度"""
    torch.manual_seed(42)

    in_features = 1024
    out_features = 1024
    batch_size = 32
    seq_len = 64

    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(layer.weight, std=0.02)

    # 模拟有 outlier channel 的激活
    # 在真实 LLM 中，某些 channel 的激活值会特别大（10x ~ 100x）
    calibration_data = torch.randn(batch_size, seq_len, in_features)
    # 制造一些 salient channels
    calibration_data[:, :, 0] *= 10   # channel 0 激活值大
    calibration_data[:, :, 5] *= 8
    calibration_data[:, :, 10] *= 6

    # 对应地，这些 channel 的权重也更"重要"
    with torch.no_grad():
        layer.weight[:, 0] *= 0.5   # 权重可能不大，但激活大
        layer.weight[:, 5] *= 0.3
        layer.weight[:, 10] *= 0.4

    # 测试数据 (与校准数据类似的分布)
    test_input = torch.randn(8, 16, in_features)
    test_input[:, :, 0] *= 10
    test_input[:, :, 5] *= 8
    test_input[:, :, 10] *= 6

    with torch.no_grad():
        ref_output = layer(test_input)

    print("=" * 85)
    print("AWQ vs RTN 精度对比")
    print(f"Layer: Linear({in_features}, {out_features})")
    print(f"Salient channels: 0 (10x), 5 (8x), 10 (6x)")
    print("=" * 85)

    for bits in [4]:
        for group_size in [128, 64]:
            print(f"\n--- {bits}-bit, group_size={group_size} ---")

            # RTN
            q_rtn, scale_rtn = symmetric_quantize(layer.weight.data, bits=bits, group_size=group_size)
            w_rtn = dequantize(q_rtn, scale_rtn, group_size=group_size)
            out_rtn = test_input.reshape(-1, in_features) @ w_rtn.T
            mse_rtn = ((ref_output.reshape(-1, out_features) - out_rtn) ** 2).mean().item()
            cos_rtn = torch.cosine_similarity(
                ref_output.flatten(), out_rtn.flatten(), dim=0
            ).item()

            # AWQ
            awq = AWQ(bits=bits, group_size=group_size)
            q_awq, q_scale, ch_scale, best_alpha = awq.quantize_layer(
                layer, calibration_data, n_grid=20
            )
            w_awq = dequantize(q_awq, q_scale, group_size=group_size) / ch_scale[None, :]
            out_awq = test_input.reshape(-1, in_features) @ w_awq.T
            mse_awq = ((ref_output.reshape(-1, out_features) - out_awq) ** 2).mean().item()
            cos_awq = torch.cosine_similarity(
                ref_output.flatten(), out_awq.flatten(), dim=0
            ).item()

            print(f"  {'方法':<12} | {'Output MSE':>12} | {'CosineSim':>10} | {'Best α':>8}")
            print(f"  {'-'*52}")
            print(f"  {'RTN':<12} | {mse_rtn:>12.2e} | {cos_rtn:>10.6f} | {'N/A':>8}")
            print(f"  {'AWQ':<12} | {mse_awq:>12.2e} | {cos_awq:>10.6f} | {best_alpha:>8.2f}")

            if mse_rtn > 0 and mse_awq > 0:
                improvement = mse_rtn / mse_awq
                print(f"  AWQ MSE 降低: {improvement:.1f}x")


def visualize_activation_aware_scaling():
    """
    可视化 AWQ 的 activation-aware scaling 效果
    """
    print("\n" + "=" * 85)
    print("AWQ Scaling 直觉解释")
    print("=" * 85)

    # 简化示例: 4 个 channel 的权重
    weight = torch.tensor([
        [0.3, 0.1, 0.5, 0.2],  # row 0
        [0.4, 0.2, 0.3, 0.1],  # row 1
    ], dtype=torch.float32)

    # 激活: channel 2 的激活值特别大
    activation = torch.tensor([
        [1.0, 1.0, 10.0, 1.0],  # token 0
        [0.8, 1.2, 8.0,  0.9],  # token 1
    ])

    print(f"权重:\n{weight}")
    print(f"\n激活 (注意 channel 2 很大):\n{activation}")

    # 原始 matmul
    ref = activation @ weight.T
    print(f"\n原始输出: {ref}")

    # RTN INT4 量化
    qmax = 7
    scale_rtn = weight.abs().amax(dim=1, keepdim=True) / qmax
    q_rtn = torch.round(weight / scale_rtn).clamp(-8, 7)
    w_rtn = q_rtn * scale_rtn
    out_rtn = activation @ w_rtn.T
    print(f"\nRTN 量化权重:\n{w_rtn}")
    print(f"RTN 输出: {out_rtn}")
    print(f"RTN 误差: {(ref - out_rtn).abs()}")

    # AWQ: 按激活 scale 调整权重
    act_scale = activation.abs().mean(dim=0)
    print(f"\n激活 scale: {act_scale}")

    alpha = 0.5
    s = act_scale.pow(alpha)
    print(f"Scaling factor (α={alpha}): {s}")

    # Scale 权重: salient channel 的权重被放大
    w_scaled = weight * s[None, :]
    print(f"\nScaled 权重 (channel 2 被放大):\n{w_scaled}")

    # 量化 scaled weight
    scale_awq = w_scaled.abs().amax(dim=1, keepdim=True) / qmax
    q_awq = torch.round(w_scaled / scale_awq).clamp(-8, 7)
    w_awq_deq = q_awq * scale_awq

    # 反向 scale
    w_awq = w_awq_deq / s[None, :]
    out_awq = activation @ w_awq.T
    print(f"AWQ 量化权重 (反 scale 后):\n{w_awq}")
    print(f"AWQ 输出: {out_awq}")
    print(f"AWQ 误差: {(ref - out_awq).abs()}")

    mse_rtn = ((ref - out_rtn) ** 2).mean().item()
    mse_awq = ((ref - out_awq) ** 2).mean().item()
    print(f"\nRTN MSE: {mse_rtn:.6f}")
    print(f"AWQ MSE: {mse_awq:.6f}")
    if mse_awq > 0:
        print(f"改善: {mse_rtn / mse_awq:.1f}x")

    print("\n关键洞察:")
    print("  - Channel 2 激活值大 (10x) → 量化误差对它影响更大")
    print("  - AWQ 通过放大 channel 2 的权重，使其量化更精细")
    print("  - 代价: 其他 channel 的量化精度略降")
    print("  - 净效果: 整体误差减小（因为 salient channel 对输出贡献更大）")


def explain_awq_vs_gptq():
    """AWQ 和 GPTQ 的对比总结"""
    print("\n" + "=" * 85)
    print("AWQ vs GPTQ 对比总结")
    print("=" * 85)

    comparison = """
    ┌──────────────────┬──────────────────────┬──────────────────────┐
    │                  │        GPTQ          │        AWQ           │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ 核心思想         │ 逐列量化             │ Activation-aware     │
    │                  │ + Hessian 误差补偿   │ per-channel scaling  │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ 需要校准数据     │ 是 (计算 Hessian)    │ 是 (计算激活 scale)  │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ 量化速度         │ 慢 (逐列处理)        │ 快 (只需搜索 α)     │
    │                  │ ~1h for 7B           │ ~10min for 7B        │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ INT4 精度        │ 很好                 │ 很好 (略优或相当)    │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ 推理 kernel 支持 │ GPTQ kernel          │ AWQ/Marlin kernel    │
    │                  │ (vLLM 支持)          │ (vLLM 推荐)         │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ vLLM 推荐度      │ ★★★★                │ ★★★★★               │
    │                  │ 用 Marlin 后端       │ Marlin 后端最佳搭配  │
    ├──────────────────┼──────────────────────┼──────────────────────┤
    │ 适合场景         │ 需要最优精度         │ 需要快速量化         │
    │                  │ 可接受长量化时间     │ + 高精度 + 快推理    │
    └──────────────────┴──────────────────────┴──────────────────────┘

    推荐选择:
    1. 首选 AWQ — 量化快、精度好、vLLM/SGLang 生态支持最好
    2. GPTQ 作为备选 — 在某些模型上精度可能更优
    3. 都不想校准 → bitsandbytes NF4 (加载即量化，无需校准数据)
    """
    print(comparison)


# ================================================================
# 主函数
# ================================================================

if __name__ == "__main__":
    print("=" * 85)
    print("03 AWQ Implementation: Activation-Aware Weight Quantization")
    print("=" * 85)

    # 1. 直觉解释
    visualize_activation_aware_scaling()

    # 2. 精度对比
    compare_awq_vs_rtn()

    # 3. AWQ vs GPTQ 总结
    explain_awq_vs_gptq()
