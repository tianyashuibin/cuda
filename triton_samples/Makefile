# Triton 需要 Python 环境，用 Makefile 管理运行
PYTHON := python3

SAMPLES := \
    01_vector_add.py \
    02_softmax.py \
    03_matmul.py \
    04_flash_attention.py \
    05_layernorm.py \
    06_quantize_matmul.py

# 运行所有示例
all: $(SAMPLES:.py=.run)

%.run: %.py
	@echo "=========================================="
	@echo "Running $<"
	@echo "=========================================="
	$(PYTHON) $<
	@echo ""

# 运行单个示例: make run-01
run-%: %.py
	$(PYTHON) $<

# 只做 import 检查（不需要 GPU）
check-env:
	$(PYTHON) -c "import triton; import torch; print(f'Triton {triton.__version__}, PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# 安装依赖
install:
	pip install triton torch

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: all check-env install clean
