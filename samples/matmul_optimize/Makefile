BUILD := build

TARGETS := \
    $(BUILD)/matmul_sharemem \
    $(BUILD)/matmul_blocksize32 \
    $(BUILD)/matmul_fp16 \
    $(BUILD)/matmul_cublas \
    $(BUILD)/matmul_tensorcore

all: $(BUILD) $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/matmul_sharemem: matmul_sharemem.cu
	nvcc -O2 -o $@ $< -lm

$(BUILD)/matmul_blocksize32: matmul_blocksize32.cu
	nvcc -O2 -o $@ $< -lm

$(BUILD)/matmul_fp16: matmul_fp16.cu
	nvcc -O2 -arch=sm_70 -o $@ $< -lm

$(BUILD)/matmul_cublas: matmul_cublas.cu
	nvcc -O2 -o $@ $< -lcublas -lm

$(BUILD)/matmul_tensorcore: matmul_tensorcore.cu
	nvcc -O2 -arch=sm_70 -o $@ $< -lm

clean:
	rm -rf $(BUILD)

.PHONY: all clean
