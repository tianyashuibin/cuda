# LLM 量化 (Quantization) 完整指南

> 从理论到实践：如何自己量化一个大模型

---

## 一、为什么需要量化

LLM 推理的核心瓶颈是 **显存带宽 (memory bandwidth)**：

```
LLaMA-7B 权重大小:
  FP32: 28 GB  → 单卡 A100 (80GB) 勉强放下，推理很慢
  FP16: 14 GB  → A100 能跑，但 batch 受限
  INT8:  7 GB  → 宽裕，可以跑更大 batch
  INT4: 3.5 GB → 单张 4090 (24GB) 轻松运行
```

**量化的本质**：用更少的 bit 表示权重（和/或激活），换取：
1. 更小的模型体积 → 更少的显存占用
2. 更少的数据搬运 → 更高的推理吞吐
3. 代价：精度损失（好的量化方法可以将损失控制到几乎无感）

---

## 二、量化基础概念

### 2.1 对称量化 vs 非对称量化

```
对称量化 (Symmetric):
  x_quant = round(x / scale)
  x_dequant = x_quant * scale
  其中 scale = max(|x|) / (2^(bits-1) - 1)
  
  特点：zero_point = 0，计算简单，适合权重

非对称量化 (Asymmetric):
  x_quant = round(x / scale) + zero_point
  x_dequant = (x_quant - zero_point) * scale
  其中 scale = (max(x) - min(x)) / (2^bits - 1)
       zero_point = round(-min(x) / scale)
  
  特点：能更好利用量化范围，适合激活值（往往不对称）
```

### 2.2 量化粒度 (Granularity)

```
Per-Tensor:    整个 tensor 共享一个 scale     → 精度最低，速度最快
Per-Channel:   每个输出通道一个 scale           → 精度适中，最常用
Per-Group:     每 group_size 个元素一个 scale   → 精度最高，INT4 必备
Per-Token:     激活的每个 token 一个 scale       → 用于 W8A8 场景

Per-Group 示例 (group_size=128):
  权重 shape: [4096, 4096]
  scale shape: [4096, 4096/128] = [4096, 32]
  每 128 个连续元素共享一个 scale
```

### 2.3 量化类型分类

| 类型 | 含义 | 代表方法 | 适用场景 |
|------|------|---------|---------|
| W8A8 | 权重 INT8, 激活 INT8 | SmoothQuant | 大 batch prefill |
| W8A16 | 权重 INT8, 激活 FP16 | 简单 RTN | 通用 |
| W4A16 | 权重 INT4, 激活 FP16 | GPTQ, AWQ, Marlin | decode 推理主流 |
| W4A4 | 权重 INT4, 激活 INT4 | QuIP# | 研究阶段 |
| FP8 | E4M3/E5M2 | NVIDIA FP8 | Hopper+ 硬件原生支持 |

---

## 三、主流量化方法

### 3.1 RTN (Round-To-Nearest) — 最简单的基线

```python
# 直接四舍五入，不做任何校准
scale = weight.abs().amax(dim=-1, keepdim=True) / 127  # per-channel
weight_int8 = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
```

**问题**：INT4 下精度崩溃严重，因为只有 16 个量化级别，直接舍入误差太大。

### 3.2 GPTQ — 逐列量化 + 误差补偿

**核心思想**：量化第 i 列时，把产生的误差补偿到还未量化的列上。

```
算法流程:
1. 用少量校准数据 (calibration data, ~128 条) 算出 Hessian 矩阵 H = 2 * X^T * X
2. 对 H 做 Cholesky 分解
3. 按列从左到右量化：
   a. 量化第 i 列: w_q[i] = quantize(w[i])
   b. 计算量化误差: δ = w[i] - dequant(w_q[i])  
   c. 把误差分摊到未量化列: w[i+1:] -= δ * H[i, i+1:] / H[i, i]
```

**优点**：INT4 per-group 量化下精度远好于 RTN
**缺点**：量化速度较慢（逐列处理），对校准数据有一定依赖

### 3.3 AWQ (Activation-Aware Quantization) — 保护重要通道

**核心思想**：不是所有权重同等重要。少数通道对应着大激活值，量化误差对它们的影响更大。

```
算法流程:
1. 用校准数据统计每个通道的激活值大小: s = activation.abs().mean(dim=0)
2. 找到 salient channels (激活值大的通道)
3. 对这些通道的权重做 per-channel scaling: w_scaled = w * s^α (α 通过搜索确定)
4. 量化 scaled weight，推理时补偿 scale
```

**优点**：不需要逐列处理（比 GPTQ 快），精度与 GPTQ 相当
**缺点**：需要搜索最优 α

### 3.4 SmoothQuant — W8A8 的关键

**核心思想**：激活的离群值 (outlier) 使量化困难。把激活的难度"转移"给权重。

```
原始: Y = X @ W
变换: Y = (X / s) @ (W * s)   其中 s = sqrt(max(|X|) / max(|W|))

效果：激活变"平滑"（easier to quantize），权重稍难量化但可以接受
```

---

## 四、自己动手量化 — 实操步骤

### 4.1 工具选择

| 工具 | 支持方法 | 推理引擎 | 推荐场景 |
|------|---------|---------|---------|
| AutoGPTQ | GPTQ | vLLM, TGI | W4A16 量化 |
| AutoAWQ | AWQ | vLLM, TGI | W4A16 量化（推荐） |
| llama.cpp | GGUF 多种格式 | llama.cpp | 本地/边缘部署 |
| NVIDIA TensorRT-LLM | FP8/INT8/INT4 | TRT-LLM | 生产部署 |
| bitsandbytes | NF4, INT8 | HuggingFace | 快速实验 |
| HQQ | Half-Quadratic Quantization | vLLM | 无需校准数据 |

### 4.2 用 AutoAWQ 量化一个模型（推荐入门）

```bash
pip install autoawq
```

```python
# awq_quantize.py — 完整的 AWQ 量化流程
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
quant_path = "./Llama-3.1-8B-AWQ"

# 1. 加载原始模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 配置量化参数
quant_config = {
    "zero_point": True,      # 使用非对称量化
    "q_group_size": 128,     # 每 128 个元素一组
    "w_bit": 4,              # INT4 量化
    "version": "GEMM",       # GEMM kernel (用于 GPU 推理)
}

# 3. 执行量化（需要校准数据，AutoAWQ 默认用 Pile 数据集）
model.quantize(tokenizer, quant_config=quant_config)

# 4. 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"量化完成，模型保存至 {quant_path}")
```

### 4.3 用 AutoGPTQ 量化

```bash
pip install auto-gptq
```

```python
# gptq_quantize.py — GPTQ 量化流程
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

model_path = "meta-llama/Llama-3.1-8B"
quant_path = "./Llama-3.1-8B-GPTQ"

# 1. 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,              # INT4
    group_size=128,      # per-group
    desc_act=False,      # False = 按顺序量化 (快); True = 按 Hessian 对角线排序 (更精确但慢)
)

# 2. 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

# 3. 准备校准数据
# GPTQ 需要校准数据来计算 Hessian
import datasets
dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
examples = [tokenizer(text, return_tensors="pt") for text in dataset["text"][:128] if len(text) > 0]

# 4. 执行量化
model.quantize(examples)

# 5. 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 4.4 用 bitsandbytes 快速量化（最简单，适合实验）

```python
# bnb_quantize.py — 加载时直接量化，无需校准
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "meta-llama/Llama-3.1-8B"

# NF4 量化配置 (QLoRA 使用的方案)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4, 比 uniform INT4 更好
    bnb_4bit_compute_dtype=torch.float16,  # 计算时用 FP16
    bnb_4bit_use_double_quant=True,        # 对 scale 再量化 (节省显存)
)

# 加载时自动量化
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 直接推理
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 4.5 用 llama.cpp 量化（本地部署）

```bash
# 1. 转换为 GGUF 格式
python convert_hf_to_gguf.py ./Llama-3.1-8B --outtype f16 --outfile llama3.1-8b-f16.gguf

# 2. 量化 (常用格式)
./llama-quantize llama3.1-8b-f16.gguf llama3.1-8b-Q4_K_M.gguf Q4_K_M

# 常用量化格式:
#   Q4_0    — 基础 INT4, 速度快但精度一般
#   Q4_K_M  — K-quant INT4 混合精度, 推荐 (精度与速度平衡)
#   Q5_K_M  — K-quant INT5, 精度更好
#   Q8_0    — INT8, 精度最好但体积大
#   Q4_K_S  — K-quant INT4 小版本, 比 Q4_K_M 小一点

# 3. 推理
./llama-cli -m llama3.1-8b-Q4_K_M.gguf -p "Hello" -n 128
```

---

## 五、量化后的推理部署

### 5.1 vLLM 加载量化模型

```python
from vllm import LLM, SamplingParams

# 加载 AWQ 量化模型
llm = LLM(
    model="./Llama-3.1-8B-AWQ",
    quantization="awq",          # 指定量化方法
    dtype="half",
    max_model_len=4096,
)

# 加载 GPTQ 量化模型
llm = LLM(
    model="./Llama-3.1-8B-GPTQ",
    quantization="gptq",
    dtype="half",
)

# 推理
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["What is quantization?"], params)
print(outputs[0].outputs[0].text)
```

### 5.2 TensorRT-LLM FP8 量化（Hopper GPU）

```python
# 用 NVIDIA modelopt 做 FP8 量化
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint

# 校准 + 量化
model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

# 导出为 TRT-LLM checkpoint
export_tensorrt_llm_checkpoint(model, "float16", export_dir="./trt_ckpt")

# 构建 TRT-LLM engine
# trtllm-build --checkpoint_dir ./trt_ckpt --output_dir ./engine ...
```

---

## 六、量化精度评估

量化后必须验证模型质量：

```python
# eval_quantized.py — 用 lm-evaluation-harness 评估
# pip install lm-eval

# 评估原始模型
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# 评估量化模型
# lm_eval --model vllm --model_args pretrained=./Llama-3.1-8B-AWQ,quantization=awq --tasks mmlu --batch_size 8
```

### 常见评估指标

| Benchmark | 测什么 | 可接受降幅 |
|-----------|-------|-----------|
| MMLU | 知识和推理 | < 1% |
| HellaSwag | 常识推理 | < 1% |
| Perplexity (PPL) | 语言建模质量 | < 0.5 |
| HumanEval | 代码生成 | < 2% |
| GSM8K | 数学推理 | < 2% |

---

## 七、INT4 权重打包格式 (面试重点)

INT4 每个元素只有 4 bit，但最小寻址单位是 byte (8 bit)，所以需要 pack：

```
打包 (Pack):
  两个 INT4 值打包到一个 INT8/UINT8 中
  packed = (w1 & 0xF) | (w2 << 4)

解包 (Unpack):
  w1 = (packed & 0xF)         # 低 4 bit
  w2 = (packed >> 4) & 0xF    # 高 4 bit
  
  // 如果是有符号 INT4 (-8 ~ 7):
  w1 = (packed << 4) >> 4     # 算术右移做符号扩展
  
在 CUDA kernel 中:
  - 一个 int32 可以存 8 个 INT4 值
  - 用 Tensor Core 计算时，需要先 unpack 到 FP16/BF16
  - Marlin kernel 的核心优化之一就是高效的 unpack + dequant pipeline
```

---

## 八、Kernel 级量化实现 (进阶)

参见本仓库的实现：
- [triton_samples/06_quantize_matmul.py](../triton_samples/06_quantize_matmul.py) — Triton W8A16 量化 matmul
- [01_ptq_basics.py](./01_ptq_basics.py) — 从零实现 PTQ 量化
- [02_gptq_impl.py](./02_gptq_impl.py) — 简化版 GPTQ 算法实现
- [03_awq_impl.py](./03_awq_impl.py) — 简化版 AWQ 算法实现
- [04_w4a16_dequant_kernel.cu](./04_w4a16_dequant_kernel.cu) — CUDA INT4 解量化 kernel

### Marlin Kernel 学习路线

```
学习顺序:
1. 先理解本仓库的 Triton W8A16 (triton_samples/06_quantize_matmul.py)
2. 再看本目录的 01_ptq_basics.py 理解量化算法本身
3. 然后看 04_w4a16_dequant_kernel.cu 理解 INT4 打包/解包
4. 最后去读 Marlin 源码 (github.com/IST-DASLab/marlin)
   - 重点: marlin_cuda_kernel.cu
   - 理解 async copy + software pipelining
   - 理解如何利用 Tensor Core 做 W4A16 计算
```

---

## 九、量化方法选择决策树

```
你要做什么？
│
├── 快速实验/Fine-tuning
│   └── bitsandbytes NF4 (QLoRA) → 最简单，加载即量化
│
├── 生产部署 (GPU)
│   ├── 有 Hopper (H100/H200)?
│   │   └── FP8 (TensorRT-LLM) → 原生硬件支持，精度最好
│   │
│   ├── 用 vLLM/SGLang 部署?
│   │   ├── AWQ W4A16 → 推荐，量化快 + 精度好 + Marlin kernel 加速
│   │   └── GPTQ W4A16 → 也可以，精度类似
│   │
│   └── 用 TensorRT-LLM?
│       └── INT4 AWQ/GPTQ → TRT-LLM 内置支持
│
├── 本地/边缘部署
│   └── llama.cpp GGUF Q4_K_M → 生态最好，CPU+GPU 混合推理
│
└── 极致压缩 (2-bit)
    └── QuIP# / AQLM → 研究阶段，适合极端显存受限场景
```

---

## 十、文件索引

| 文件 | 内容 |
|------|------|
| `01_ptq_basics.py` | 从零实现 PTQ：对称/非对称量化、per-channel/per-group、精度对比 |
| `02_gptq_impl.py` | 简化版 GPTQ：逐列量化 + Hessian 误差补偿 |
| `03_awq_impl.py` | 简化版 AWQ：activation-aware scaling |
| `04_w4a16_dequant_kernel.cu` | CUDA kernel：INT4 打包解包 + dequantize |
