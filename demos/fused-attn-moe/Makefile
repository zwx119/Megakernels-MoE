# Fused Attention + MoE Megakernel Makefile
# Compiler
NVCC?=nvcc

ifndef GPU
GPU=H100
endif

TARGET=mk_fused_attn_moe
SRC=main.cu

# Python version configuration
ifndef PYTHON_VERSION
PYTHON_VERSION=3.13
endif

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
# 兼容性补丁：为不支持 std::bit_cast 的旧 GCC 提供 polyfill
NVCCFLAGS+= -include $(CURDIR)/compat.cuh
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython${PYTHON_VERSION}

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else ifeq ($(GPU),H20)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)

.PHONY: all clean
