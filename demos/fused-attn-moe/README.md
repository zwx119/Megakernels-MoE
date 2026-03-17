# 融合 Attention + MoE Megakernel (Warp Specialization)

## 核心思想

在大模型推理的 **decode 阶段**（低延迟场景，batch size 较小），Flash Attention 和 MoE 计算各自都无法将 SM 的 Tensor Core 完全利用。

**解决方案**：通过 warp specialization，在同一个 SM 内让一部分 warp 计算当前 token 的 attention，同时让另一部分 warp 计算**已经完成 attention 的历史 token** 的 MoE，从而提高 Tensor Core 利用率。

```
时间线示意：
SM 0: [=== Attention (Token T) ===] --signal--> 
      [=== MoE (Token T-1) ===]    <-- 与上面重叠执行

单个 SM 的 warp 分配：
┌──────────────────────────────────────────┐
│ Warp 0~7:  Attention Consumer Warps      │ ← Flash Attention decode
│ Warp 8~15: MoE Consumer Warps            │ ← MoE Expert MatVec
│ Warp 16:   Loader (TMA)                  │ ← 加载 KV cache + MoE 权重
│ Warp 17:   Storer                        │ ← 存储结果
│ Warp 18:   Launcher                      │ ← Tensor Core 初始化
│ Warp 19:   Controller                    │ ← 指令调度
└──────────────────────────────────────────┘
```

## 与 MegaBlocks 的区别

| 特性 | 本方案 (Fused Attn+MoE) | MegaBlocks |
|------|------------------------|------------|
| 并行级别 | SM 内 warp 级 | SM 间 block 级 |
| 融合方式 | 同一 SM 同时执行 Attn + MoE | 不同 SM 处理不同 expert |
| 目标场景 | 低延迟 decode (小 batch) | 高吞吐 (大 batch) |
| 稀疏性处理 | Token 级 routing | Block-sparse matmul |
| 框架基础 | Megakernel VM | Triton |

## 技术架构

### 基于 Megakernel 框架

完全复用 Megakernel 框架的 `default_config`：
- **16 个 consumer warps**：在 `consumer::run` 内部按 `warpid()` 分流
- **4 个 infrastructure warps**：loader/storer/launcher/controller
- **13 页共享内存**（每页 16KB）：attention 用 2 页，MoE 权重用 8 页（2-stage pipeline）
- **32 个动态信号量**：attention 用 15 个，MoE 用 9 个

### 同步机制

```
1. Attention warps 完成 → atomicAdd(attn_done_barrier[token_idx])
2. MoE warps 自旋等待 → while(attn_done_barrier[moe_token_idx] < 1)
3. 共享内存信号量 → TMA 数据传输协调
4. Global barriers (Bar) → 跨 SM 的 op 间依赖
```

### 指令格式

每条指令编码**同时进行**的 attention 和 MoE 任务：

```
[0]  opcode = 9
[1]  layer_idx
[2]  kv_head_idx        (attention)
[3]  num_partials        (attention 分区数)
[4]  partial_idx         (当前分区)
[5]  moe_token_idx       (-1=无 MoE)
[6]  moe_expert_idx      (top-k expert 索引)
[7]  moe_weight_type     (0=up, 1=gate, 2=down)
[8]  moe_start_block
[9]  moe_end_block
[10] moe_reduction_block
```

## 文件结构

```
demos/fused-attn-moe/
├── fused_attn_moe.cuh    # 头文件：全局定义、模型配置、globals 结构体
├── fused_attn_moe.cu     # 核心实现：controller/loader/launcher/consumer/storer
├── main.cu               # 入口文件：模板实例化 + pybind11 绑定
├── Makefile               # 编译配置
└── README.md              # 本文件

megakernels/demos/fused/
├── instructions.py        # Python 指令定义
├── scheduler.py           # DAG 调度器
├── python_vm.py           # Python 参考实现
├── mk.py                  # MK 解释器桥接
└── test_fused.py          # 测试脚本
```

## 编译与运行

```bash
# 设置环境变量
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens
export MEGAKERNELS_ROOT=/path/to/Megakernels

# 编译 (H100)
cd demos/fused-attn-moe
make GPU=H100

# 运行测试
cd ../../
python -m megakernels.demos.fused.test_fused
```

## 性能预期

### 适用场景
- Decode 阶段，batch_size = 1~8
- Sequence length 较长（KV cache 访存占比高）
- MoE 模型（Mixtral, DeepSeek 等）

### 预期收益
- **Tensor Core 利用率**: 从 ~20%（纯 attention decode）提升到 ~40%+
- **端到端延迟**: 减少 15-30%（通过 overlap 消除 MoE 的等待时间）

### 局限性
- Attention warp 只用了 1 个（warp 1~7 空闲），适合 decode 但不适合 prefill
- MoE 权重使用 2-stage pipeline（受限于 13 页共享内存），可能有 bubble
- 需要 token 间有依赖关系才能 overlap（第一个 token 无 MoE overlap）
