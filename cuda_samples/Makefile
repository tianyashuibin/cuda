BUILD := build

# -lineinfo: 保留源码行号信息，Nsight Compute 可将性能数据对应到具体代码行
# -O2:       保持优化，profiling 结果才反映真实性能（-G 会禁用优化，不适合 profiling）
NVCCFLAGS := -O2 -lineinfo

TARGETS := \
    $(BUILD)/cuda_graph_demo \
    $(BUILD)/cuda_stream_demo \
    $(BUILD)/vector_add

all: $(BUILD) $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/cuda_graph_demo: cuda_graph_demo.cu
	nvcc $(NVCCFLAGS) -o $@ $< -lm

$(BUILD)/cuda_stream_demo: cuda_stream_demo.cu
	nvcc $(NVCCFLAGS) -o $@ $< -lm

$(BUILD)/vector_add: vector_add.cu
	nvcc $(NVCCFLAGS) -o $@ $< -lm

clean:
	rm -rf $(BUILD)

.PHONY: all clean

