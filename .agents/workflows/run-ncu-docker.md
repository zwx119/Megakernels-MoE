---
description: 在 Docker 中运行 Nsight Compute Profiling 配置流程
---

# 在 Docker 中运行 Nsight Compute Profiling

为了在 Docker 中运行 Nsight Compute (ncu)，需要特殊的权限配置以允许 Docker 访问 GPU 的性能计数器。

## 1. 启动 Docker 容器 (关键设置)

当你在服务器上启动 Docker 容器时，**必须**加上 `--privileged` 标志，并且最好开启 host IPC 以防止 shared memory 问题。

```bash
docker run -it --gpus all \
    --privileged \
    --ipc=host \
    -v /home/user01/.cache/modelscope:/home/user01/.cache/modelscope \
    -v $(pwd):/workspace/Megakernels \
    nvcr.io/nvidia/pytorch:24.02-py3 \
    /bin/bash
```

*注意：`nvcr.io/nvidia/pytorch:24.02-py3` 已经自带了 nvcc、ncu 等必要的开发工具。*

## 2. 容器内环境配置

进入 Docker 容器后，重新配置 Megakernels 的依赖：

```bash
cd /workspace/Megakernels
git submodule update --init --recursive

# 安装依赖
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

# 编译 Megakernel (H100 为例)
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.10  # 根据镜像内的 Python 版本调整 (python --version)
export GPU=H100

cd demos/low-latency-llama
make
cd ../..
```

## 3. 运行 Nsight Compute (PyTorch 模式)

在 Docker 内使用 ncu 时，如果遇到权限问题，可以尝试以下修复（如果启动时加了 `--privileged` 通常不需要）：
```bash
# 如果 ncu 报错没有权限，可能需要在 docker 外的 host 执行:
# sudo modprobe nvidia # 确保驱动加载
# sudo chmod 666 /dev/nvidiactl /dev/nvidia0 # 调整设备权限
```

运行 Profiler，因为是 PyTorch 模式，这会输出每个 kernel 的详细信息：

```bash
ncu --set full \
    --target-processes all \
    --kernel-name-base demangled \
    -o /tmp/ncu_pytorch_baseline \
    python megakernels/scripts/generate.py \
      mode=torch \
      model="/home/user01/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct" \
      prompt="tell me a funny joke about cookies" \
      num_warmup=0 num_iters=1 ntok=10 tokens=False
```

## 4. (重要) Megakernel 内部耗时获取
NCU 只能看到整体 Kernel 时间，如果要看 Megakernel 内部 FA 和 MLP 各花了多少时间，**这部分不需要 ncu**，直接跑即可：

```bash
# 修改代码开启 Timing
sed -i 's/TIMING_RECORD_ENABLED = false/TIMING_RECORD_ENABLED = true/' include/config.cuh
cd demos/low-latency-llama && make && cd ../..

# 跑 benchmark 并导出
python megakernels/scripts/diff_test.py \
    model="/home/user01/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct" \
    outfile=/tmp/mk_timing.pkl full
```
