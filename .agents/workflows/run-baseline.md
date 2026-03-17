---
description: 在 H100 上编译并运行 Megakernels baseline benchmark
---

# 在 H100 上运行 Baseline Benchmark

## 前提
- 已 SSH 到有 H100 GPU 的服务器
- 代码已 clone 到服务器上

## 步骤

### 1. 环境准备
// turbo
```bash
cd /path/to/Megakernels  # 替换为你的实际路径
git submodule update --init --recursive
```

### 2. 安装依赖
```bash
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

### 3. 编译 Megakernel
```bash
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12  # 按实际版本调整
export GPU=H100
cd demos/low-latency-llama
make
cd ../..
```

### 4. 运行 Benchmark
```bash
# 端到端生成性能测试 (MK 模式)
python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100

# 对比 PyTorch baseline
python megakernels/scripts/generate.py mode=pytorch prompt="tell me a funny joke about cookies" ntok=100

# 引擎对比测试 (如果 bench_engines.py 支持)
python megakernels/scripts/bench_engines.py
```

### 5. 开启 per-op timing（可选）
在 `include/config.cuh` 中修改：
```cpp
static constexpr bool TIMING_RECORD_ENABLED = true;
```
然后重新编译 (`cd demos/low-latency-llama && make && cd ../..`) 再跑 benchmark。
