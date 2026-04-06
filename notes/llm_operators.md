# LLM 常用算子整理

## 1. 线性变换类

| 算子 | 说明 | 计算特征 |
|------|------|----------|
| **GEMM** (矩阵乘法) | QKV 投影、FFN 层、输出投影 | 计算密集型，占总计算量 ~60-70% |
| **Batched GEMM** | Multi-Head Attention 中多头并行矩阵乘 | 多个小 GEMM 并行 |

## 2. Attention 相关

| 算子 | 说明 | 计算特征 |
|------|------|----------|
| **Scaled Dot-Product Attention** | `softmax(QK^T / sqrt(d)) * V` | 序列长度增大时显存瓶颈 |
| **FlashAttention** | 融合 tiling 的 attention | 减少 HBM 访问，IO-aware |
| **KV Cache** | 推理时缓存 K/V，避免重复计算 | 显存密集型 |
| **RoPE** (旋转位置编码) | 对 Q/K 施加旋转矩阵 | 逐元素操作 |
| **ALiBi** | 注意力线性偏置 | 逐元素加偏置 |

## 3. 归一化类

| 算子 | 说明 | 计算特征 |
|------|------|----------|
| **LayerNorm** | 每个 token 维度归一化 | 访存密集型，需 reduce 操作 |
| **RMSNorm** | 省去均值计算的简化版 LayerNorm | LLaMA 系列常用，比 LayerNorm 快 |

## 4. 激活函数类

| 算子 | 说明 | 使用场景 |
|------|------|----------|
| **GeLU** | 高斯误差线性单元 | GPT 系列 FFN |
| **SiLU / Swish** | `x * sigmoid(x)` | LLaMA FFN |
| **SwiGLU** | `SiLU(xW1) * (xW2)` — 门控线性单元 | LLaMA / Mistral FFN，目前主流 |

## 5. Softmax 类

| 算子 | 说明 | 计算特征 |
|------|------|----------|
| **Softmax** | Attention 分数归一化 | 需要 reduce (max, sum)，数值稳定性关键 |
| **Online Softmax** | 单 pass 流式 softmax | FlashAttention 核心技巧 |

## 6. Embedding / 采样类

| 算子 | 说明 | 计算特征 |
|------|------|----------|
| **Embedding Lookup** | token → 向量查表 | 访存密集型，不规则访问 |
| **Top-k / Top-p Sampling** | 推理时从 logits 采样 | 排序 + 采样 |
| **Cross Entropy Loss** | 训练时计算损失 | softmax + log + reduce |

## 7. 通信类 (分布式)

| 算子 | 说明 | 使用场景 |
|------|------|----------|
| **AllReduce** | 多卡梯度聚合 | 数据并行 |
| **AllGather / ReduceScatter** | 张量切片通信 | 张量并行 |
| **P2P Send/Recv** | 流水线阶段间传输 | 流水线并行 |

## 8. 算子融合 (Kernel Fusion) 常见模式

| 融合模式 | 说明 |
|----------|------|
| **GEMM + Bias + Activation** | 矩阵乘后直接加偏置和激活 |
| **QKV Projection 融合** | 三个投影合成一个大 GEMM |
| **Attention 融合** (FlashAttention) | QK^T → scale → mask → softmax → V 一次完成 |
| **LayerNorm/RMSNorm + 残差连接** | 归一化和 add 融合 |
| **SwiGLU 融合** | gate 和 up projection 融合 |

## 9. 性能瓶颈分布 (典型 Transformer)

```
计算密集型 (Compute-bound)          访存密集型 (Memory-bound)
┌──────────────────────┐          ┌──────────────────────┐
│  GEMM (~70% FLOPs)   │          │  Softmax             │
│  Batched GEMM        │          │  LayerNorm / RMSNorm │
│                      │          │  激活函数 (element-wise)│
│                      │          │  Residual Add         │
│                      │          │  Embedding Lookup     │
│                      │          │  RoPE                 │
└──────────────────────┘          └──────────────────────┘
优化重点: Tiling, Tensor Core       优化重点: Kernel Fusion, 减少显存搬运
```

## 10. CUDA 优化优先级建议

1. **GEMM** — 计算量最大，优化收益最高（shared memory tiling → register tiling → Tensor Core）
2. **FlashAttention** — Attention 融合，减少显存搬运，长序列必备
3. **RMSNorm / LayerNorm** — 访存密集，融合残差连接可显著提速
4. **SwiGLU / 激活融合** — 与 GEMM 融合减少 kernel launch 开销
5. **Softmax** — 数值稳定性 + reduce 优化
6. **KV Cache 管理** — 推理场景核心瓶颈（PagedAttention 等）
