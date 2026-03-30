# PyTorch 常用算子与 Kernel Launch 对应关系

## 线性代数

| 算子 | Kernel Launch 次数 | 底层库 | 备注 |
|------|-------------------|--------|------|
| `torch.matmul` / `torch.mm` | 1 | cuBLAS | 矩阵乘法 |
| `torch.bmm` | 1 | cuBLAS | batch 矩阵乘法 |
| `torch.addmm` | 1 | cuBLAS | `C = beta*C + alpha*(A@B)`，加法与乘法融合 |
| `torch.mv` | 1 | cuBLAS | 矩阵向量乘法 |
| `torch.dot` | 1 | cuBLAS | 向量内积 |

## 激活函数

| 算子 | Kernel Launch 次数 | 备注 |
|------|-------------------|------|
| `F.relu` | 1 | |
| `F.gelu` | 1 | |
| `F.silu` | 1 | |
| `F.sigmoid` | 1 | |
| `F.tanh` | 1 | |
| `F.softmax` | 2~3 | max → sum → div，分步计算 |

## 归一化

| 算子 | Kernel Launch 次数 | 备注 |
|------|-------------------|------|
| `F.layer_norm` | 2~3 | 均值、方差、归一化分步 |
| `F.batch_norm` | 2~3 | 训练时含统计量更新 |
| `F.rms_norm` | 1~2 | LLaMA 等模型使用 |

## 卷积

| 算子 | Kernel Launch 次数 | 底层库 | 备注 |
|------|-------------------|--------|------|
| `F.conv2d` | 1 | cuDNN | 内部自动选择最优算法 |
| `F.conv1d` | 1 | cuDNN | |
| `F.conv_transpose2d` | 1 | cuDNN | 转置卷积 |

## Element-wise 运算

| 算子 | Kernel Launch 次数 | 备注 |
|------|-------------------|------|
| `x + y` / `x - y` | 1 | |
| `x * y` / `x / y` | 1 | |
| `x ** 2` | 1 | |
| `torch.sqrt(x)` | 1 | |
| `torch.exp(x)` | 1 | |
| `torch.log(x)` | 1 | |
| `torch.clamp(x)` | 1 | |

## Reduction

| 算子 | Kernel Launch 次数 | 备注 |
|------|-------------------|------|
| `x.sum()` | 1~2 | 大数组可能分两阶段 |
| `x.mean()` | 1~2 | |
| `x.max()` / `x.min()` | 1 | |
| `x.argmax()` | 1 | |
| `x.norm()` | 1~2 | |

## 内存操作

| 算子 | Kernel Launch 次数 | 备注 |
|------|-------------------|------|
| `x.reshape` / `x.view` | 0 | 只改元数据，不动数据 |
| `x.transpose` / `x.permute` | 0 | 只改 stride，惰性操作 |
| `x.contiguous()` | 1 | 数据不连续时才触发拷贝 |
| `torch.cat` | 1 | |
| `torch.stack` | 1 | |
| `x[mask]` (masked select) | 1~2 | |

## 高层复合算子（拆解后的实际 launch 数）

| 算子 | 朴素实现 | 优化实现 |
|------|---------|---------|
| `F.scaled_dot_product_attention` | 6~8次 | 1次（Flash Attention）|
| `F.linear` | 1~2次 | 1次（addmm 融合）|
| `F.dropout` | 2次（生成 mask + 应用）| 1次（融合）|
| Transformer layer | 50~100次 | 5~10次（compile 后）|

## PyTorch 的算子融合机制

| 机制 | 说明 |
|------|------|
| `torch.compile` | 编译时分析计算图，自动融合 element-wise 算子 |
| `torch.compile` + Triton | 生成自定义融合 kernel |
| cuDNN / cuBLAS 内置融合 | `addmm`、bias add 等底层已融合 |
| CUDA Graph | 消除整个 forward pass 的 launch overhead |
| Flash Attention | 手写融合 kernel，消除 O(N²) 显存读写 |

## 典型 Transformer Layer 的 Launch 次数估算

```
未优化：         ~50~100 次 kernel launch
torch.compile：  ~10~20 次
+ Flash Attn：   ~5~10 次
+ CUDA Graph：   launch overhead 全部消除
```
