NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -lineinfo -I.

.PHONY: all clean

all: plain_core engineered_core

plain_core: plain/plain_main.cu plain/plain_build.cu plain/plain_search.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

engineered_core: engineered/engineered_main.cu engineered/engineered_plan.cu engineered/engineered_build.cu engineered/engineered_search.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f plain_core engineered_core

