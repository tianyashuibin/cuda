# CUDA Graph

## 解决的问题

每次 kernel launch，CPU 都要经历完整的调度流程：

```
参数验证 → 命令编码 → 写入 command buffer → 通知 driver → 提交到 GPU 硬件队列
```

每次约 5~20 μs。一个 Transformer forward pass 有几百个 kernel，累计几毫秒全花在 CPU 调度上，GPU 反而在等待。

---

## 原理

本质是把"动态生成命令"变成"预编译命令包"。

### 普通 kernel launch 的完整路径

```
每次 kernel launch，CPU 做：
  1. 参数验证（指针是否合法、grid/block 是否越界...）
  2. 将命令编码成 GPU 能理解的二进制格式
  3. 将命令写入 command buffer
  4. 通知 GPU driver 有新命令
  5. GPU driver 将命令提交到 GPU 硬件队列

这 5 步每次都重复，花 5~20 μs
```

### 三个阶段

**1. 录制阶段（capture）**

`cudaStreamBeginCapture` 之后，所有 kernel launch 不真正执行，而是被拦截，记录成图的节点：

```
节点 A（kernel + 参数 + grid/block）
  ↓
节点 B（kernel + 参数 + grid/block）
  ↓
节点 C（memcpy + src + dst + size）
```

**2. 实例化阶段（instantiate）**

`cudaGraphInstantiate` 把这张图编译成预先验证、预先编码好的 command buffer：

```
[验证 ✓][编码好的命令序列: A → B → C → ...]
```

这步有开销，但只做一次。

**3. 回放阶段（launch）**

`cudaGraphLaunch` 只做一件事：把预编译好的 command buffer 直接提交给 GPU 硬件队列。

```
跳过了：参数验证、命令编码、逐个提交
CPU 开销从 N × 10μs 降到 ~1μs
```

---

## 执行对比

```
普通执行：
  CPU: launch A → launch B → launch C → launch D ...（每次都有调度开销）
  GPU:            [A]         [B]         [C]         [D]

CUDA Graph：
  CPU: launch graph（一次提交）
  GPU: [A][B][C][D]...（连续执行，无等待）
```

---

## 使用方式

```cpp
// 1. 录制
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<grid, block, 0, stream>>>(...);
kernelB<<<grid, block, 0, stream>>>(...);
kernelC<<<grid, block, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);

// 2. 实例化（编译图，一次性开销）
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 3. 之后每次推理只需一行
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);
```

---

## 限制

| 限制 | 原因 |
|------|------|
| kernel 指针和参数不能变 | command buffer 已硬编码 |
| batch size 不能变 | grid/block 维度固定在图里 |
| 动态控制流不能用 | if/else 在录制时已固定 |

**为什么 batch size 变了就不能用同一张图：**

command buffer 里的命令已经把 grid 大小硬编码：
```
grid = (batch_size / 32, hidden / 32)   ← batch size 变了这里就错了
kernel_ptr = 0x7f3a...                  ← 数据指针也固定了
```

---

## 工业实践：vLLM 的做法

vLLM 针对每个可能的 batch size 各录制一张图：

```
graph[1]  ← batch size = 1 时使用
graph[2]  ← batch size = 2 时使用
graph[4]  ← batch size = 4 时使用
graph[8]  ← batch size = 8 时使用
...
```

推理时根据实际 batch size 查表，选对应的图执行。

---

## 类比

```
普通执行：每次做饭都现场买菜、洗菜、切菜、炒菜
CUDA Graph：提前做好"料理包"，要吃时直接加热
```

---

## 与算子融合的区别

| | 算子融合 | CUDA Graph |
|--|---------|-----------|
| 解决的问题 | 显存读写次数过多 + kernel 数量过多 | CPU launch overhead |
| 手段 | 合并 kernel，中间结果留在寄存器/shared memory | 预编译命令包，一次提交所有 launch |
| 主要收益 | 减少显存带宽消耗 | 减少 CPU 调度延迟 |
| 改变了什么 | 计算方式 | 启动方式 |
| 典型场景 | 训练/推理的计算密集路径 | 推理的重复 forward pass |

两者互补，生产环境通常同时使用。
