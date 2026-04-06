BUILD := build

# -lineinfo: 保留源码行号信息，Nsight Compute 可将性能数据对应到具体代码行
# -O2:       保持优化，profiling 结果才反映真实性能（-G 会禁用优化，不适合 profiling）
NVCCFLAGS := -O2 -lineinfo

TARGETS := \
    $(BUILD)/matmul_sharemem \
    $(BUILD)/matmul_reg_tiling \
    $(BUILD)/matmul_fp16 \
    $(BUILD)/matmul_cublas \
    $(BUILD)/matmul_tensorcore

all: $(BUILD) $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/matmul_sharemem: matmul_sharemem.cu
	nvcc $(NVCCFLAGS) -o $@ $< -lm

$(BUILD)/matmul_reg_tiling: matmul_reg_tiling.cu
	nvcc $(NVCCFLAGS) -o $@ $< -lm

$(BUILD)/matmul_fp16: matmul_fp16.cu
	nvcc $(NVCCFLAGS) -arch=sm_70 -o $@ $< -lm

$(BUILD)/matmul_cublas: matmul_cublas.cu
	nvcc $(NVCCFLAGS) -o $@ $< -lcublas -lm

$(BUILD)/matmul_tensorcore: matmul_tensorcore.cu
	nvcc $(NVCCFLAGS) -arch=sm_70 -o $@ $< -lm

clean:
	rm -rf $(BUILD)

.PHONY: all clean

