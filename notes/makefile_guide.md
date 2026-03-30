# Makefile 使用说明

## 基本概念

Makefile 是一个构建脚本，告诉 `make` 命令如何编译项目。核心思想是：

```
目标: 依赖
	命令（必须用 Tab 缩进，不能用空格）
```

---

## 本项目的 Makefile 结构

```
cuda/
  Makefile                      ← 顶层，统一入口，委托给子目录
  samples/
    Makefile                    ← 编译 cuda_graph_demo、cuda_stream_demo
    matmul_optimize/
      Makefile                  ← 编译所有 matmul 系列
```

---

## 常用命令

### 编译

```bash
# 在任意目录下执行 make，编译该目录下的所有目标
make

# 只编译某一个目标
make build/matmul_cublas

# 在根目录编译所有子项目（会递归进入每个子目录）
cd cuda/
make
```

### 清理编译产物

```bash
# 删除 build/ 目录及其中所有编译产物
make clean
```

### 指定子目录编译

```bash
# 不进入子目录，直接在根目录指定
make -C samples
make -C samples/matmul_optimize
```

---

## 读懂本项目的 Makefile

### 顶层 Makefile（cuda/Makefile）

```makefile
SUBDIRS := \
    samples \
    samples/matmul_optimize

all:
	for dir in $(SUBDIRS); do \
	    $(MAKE) -C $$dir; \     # 进入每个子目录执行 make
	done

clean:
	for dir in $(SUBDIRS); do \
	    $(MAKE) -C $$dir clean; \
	done
```

- `SUBDIRS` — 定义变量，列出所有子目录
- `$(MAKE) -C $$dir` — 进入 `$$dir` 目录执行 make（`$$` 是 Makefile 里 `$` 的转义）
- `all` 和 `clean` — 两个目标，`make` 默认执行第一个目标（即 `all`）

### 子目录 Makefile（samples/matmul_optimize/Makefile）

```makefile
BUILD := build           # 定义变量，编译产物输出目录

TARGETS := \             # 所有要生成的可执行文件
    $(BUILD)/matmul_sharemem \
    $(BUILD)/matmul_cublas \
    ...

all: $(BUILD) $(TARGETS) # all 依赖 build目录 和所有 TARGETS

$(BUILD):                # 如果 build/ 不存在就创建
	mkdir -p $(BUILD)

$(BUILD)/matmul_cublas: matmul_cublas.cu    # 目标依赖源文件
	nvcc -O2 -o $@ $< -lcublas -lm          # $@ = 目标文件名，$< = 第一个依赖文件

clean:
	rm -rf $(BUILD)
```

**常用自动变量：**

| 变量 | 含义 | 示例 |
|------|------|------|
| `$@` | 当前目标的文件名 | `build/matmul_cublas` |
| `$<` | 第一个依赖文件 | `matmul_cublas.cu` |
| `$^` | 所有依赖文件 | 多个源文件时用 |

---

## make 的增量编译

make 会比较目标文件和依赖文件的**修改时间**：

```
如果 build/matmul_cublas 比 matmul_cublas.cu 新 → 跳过，不重新编译
如果 matmul_cublas.cu 更新了                   → 重新编译
如果 build/matmul_cublas 不存在                → 编译
```

所以修改了某个 `.cu` 文件后，`make` 只会重新编译那一个文件，其他不变的不会重编。

---

## 在 Colab 上的完整流程

```bash
# 挂载 Google Drive
# （在 notebook cell 里执行）
from google.colab import drive
drive.mount('/content/drive')

# 进入项目根目录
cd /content/drive/MyDrive/cuda

# 编译所有
make

# 运行某个程序
./samples/build/cuda_graph_demo
./samples/matmul_optimize/build/matmul_cublas

# 清理所有编译产物
make clean
```

---

## .gitignore 配合

编译产物放在 `build/` 目录后，`.gitignore` 只需一行：

```gitignore
build/
```

所有 `build/` 目录（包括子目录里的）都会被忽略，不会被提交到 git。
