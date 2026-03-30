BUILD := build

TARGETS := \
    $(BUILD)/cuda_graph_demo \
    $(BUILD)/cuda_stream_demo

all: $(BUILD) $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/cuda_graph_demo: cuda_graph_demo.cu
	nvcc -O2 -o $@ $< -lm

$(BUILD)/cuda_stream_demo: cuda_stream_demo.cu
	nvcc -O2 -o $@ $< -lm

clean:
	rm -rf $(BUILD)

.PHONY: all clean
