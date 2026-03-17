#!/usr/bin/env python3
"""
============================================================================
  融合 Attention + MoE Megakernel — 端到端 CUDA 性能测试
============================================================================

【功能】
  1. 正确性验证：对比 CUDA kernel 输出 vs PyTorch 参考实现
  2. 绝对性能：测量 fused kernel 的 forward 延迟 (µs) 和 tokens/s
  3. Baseline 对比：sequential attention + MoE 的延迟
  4. 加速比分析：fused vs sequential

【使用方法】
  # 在 GPU 机器上运行（需要先编译好 mk_fused_attn_moe.so）
  cd Megakernels
  python -m megakernels.demos.fused.bench_fused

  # 自定义参数
  python -m megakernels.demos.fused.bench_fused \
      --seq-len 512 --batch-size 1 --num-layers 2 --warmup 20 --iters 100

  # 只跑性能（跳过正确性）
  python -m megakernels.demos.fused.bench_fused --skip-correctness

【依赖】
  - 编译好的 mk_fused_attn_moe.cpython-*.so (在 demos/fused-attn-moe/ 下)
  - PyTorch with CUDA
  - einops
"""

import argparse
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import torch
import torch.nn.functional as F


# ============================================================================
# 编译时常量
# ============================================================================

# main.cu 中通过 FUSED_SM_COUNT 宏或默认值决定
# kernel 的 grid() 返回 dim3(sm_count)，指令/timing tensor 必须用编译时 SM count
# get_worker_id() 返回 smid（硬件 SM ID），所以必须与实际硬件 SM 数匹配！
# 设为 0 表示运行时自动用硬件 SM count
COMPILED_SM_COUNT = 0  # 0 = 自动检测


# ============================================================================
# 模型配置
# ============================================================================

class ModelConfig:
    """模拟 Llama-1B MoE 模型配置，与编译时的宏保持一致。"""
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=64,
        num_hidden_layers=2,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128256,
        rms_norm_eps=1e-5,
        kv_block_size=16,
        matvec_block_size=16,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.kv_block_size = kv_block_size
        self.matvec_block_size = matvec_block_size


# ============================================================================
# Tensor 创建工具
# ============================================================================

def create_model_tensors(config: ModelConfig, seq_len: int, kernel_sm_count: int,
                         batch_size: int = 1, device="cuda:0"):
    """
    创建所有模型 tensor，模拟真实推理场景。
    所有 tensor 形状与 fused_globals_t 中的 gl<> 类型完全对齐。

    pyutils.cuh 的 from_object<GL> 会将 tensor 左填充 1 到 4D 后检查：
      - gl<T, B, D, R, C> 中 B/D/R/C > 0 的维度必须精确匹配
      - -1 表示动态维度，不检查

    【关键规则】
      - gl<T, 1, ...> 的 batch=1 → 用 3D 或更低维 tensor（自动填充 batch=1）
      - 如果传 4D tensor，第一维就是 batch，必须为 1
      - gl<T, -1, -1, -1, C> 全动态 → 4D tensor 直接使用
    """
    dtype = torch.bfloat16
    num_kv_blocks = math.ceil(seq_len / config.kv_block_size)
    total_heads = config.num_attention_heads + config.num_key_value_heads * 2
    num_le = config.num_hidden_layers * config.num_experts

    # 编译时 SM count 用于 intermediates 的尺寸
    lse_col_padded = ((kernel_sm_count + 15) // 16) * 16

    tensors = {}

    # ---- VM 基础设施 ----
    # barriers: gl<uint, 1, -1, -1, N>  N = total_heads = 48
    # → 3D tensor [num_layers, 12, 48] → padded to [1, num_layers, 12, 48]  b=1 ✓
    tensors['barriers'] = torch.zeros(
        config.num_hidden_layers, 12, total_heads,
        dtype=torch.int32, device=device,
    )

    # ---- QKV/O 权重 ----
    # weights_t = gl<bf16, 1, -1, -1, hidden_dim>
    # → 3D tensor [num_layers, out_features, hidden_dim]
    # → padded to [1, num_layers, out_features, hidden_dim]  b=1 ✓
    qkv_out_features = total_heads * config.head_dim  # (32+8+8)*64 = 3072
    tensors['qkv_weights'] = torch.randn(
        config.num_hidden_layers, qkv_out_features, config.hidden_size,
        device=device, dtype=dtype,
    ) * 0.01

    # norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim>
    # → 2D tensor [num_layers, hidden_dim] → padded to [1, 1, num_layers, hidden_dim]
    #   b=1 ✓, d=1 ✓
    tensors['attn_norm_weights'] = torch.ones(
        config.num_hidden_layers, config.hidden_size,
        device=device, dtype=dtype,
    )

    # o_weights: weights_t = gl<bf16, 1, -1, -1, hidden_dim>
    o_out_features = config.num_attention_heads * config.head_dim  # 32*64 = 2048
    tensors['o_weights'] = torch.randn(
        config.num_hidden_layers, o_out_features, config.hidden_size,
        device=device, dtype=dtype,
    ) * 0.01

    tensors['mlp_norm_weights'] = torch.ones(
        config.num_hidden_layers, config.hidden_size,
        device=device, dtype=dtype,
    )

    # ---- KV cache ----
    # kv_cache_t = gl<bf16, -1, -1, -1, head_dim>  全动态
    # → 4D tensor [num_layers * num_kv_blocks, num_kv_heads, kv_block_size, head_dim]
    tensors['k_cache'] = torch.randn(
        config.num_hidden_layers * num_kv_blocks,
        config.num_key_value_heads, config.kv_block_size, config.head_dim,
        device=device, dtype=dtype,
    ) * 0.1
    tensors['v_cache'] = torch.randn(
        config.num_hidden_layers * num_kv_blocks,
        config.num_key_value_heads, config.kv_block_size, config.head_dim,
        device=device, dtype=dtype,
    ) * 0.1

    # ---- RoPE ----
    # rope_table_t = gl<float, 1, 1, -1, head_dim=64>
    # → 2D tensor [seq_len, head_dim] → padded to [1, 1, seq_len, head_dim]
    tensors['rope_cos'] = torch.ones(
        seq_len + 16, config.head_dim,
        device=device, dtype=torch.float32,
    )
    tensors['rope_sin'] = torch.zeros(
        seq_len + 16, config.head_dim,
        device=device, dtype=torch.float32,
    )

    # ---- Activation 缓冲区 ----
    # activations_t = gl<bf16, 1, 1, 1, hidden_dim>  所有维度固定
    # → 1D tensor [hidden_dim] → padded to [1, 1, 1, hidden_dim]
    tensors['hidden_states'] = torch.randn(
        config.hidden_size, device=device, dtype=dtype,
    ) * 0.1
    tensors['q_post_rope'] = torch.randn(
        config.hidden_size, device=device, dtype=dtype,
    ) * 0.1
    tensors['attn_out'] = torch.zeros(
        config.hidden_size, device=device, dtype=dtype,
    )

    # attn_lse_intermediates_t = gl<float, 1, 1, num_attention_heads=32, -1>
    # → 2D tensor [32, lse_col_padded] → padded to [1, 1, 32, lse_col_padded]
    #   b=1 ✓, d=1 ✓, r=32 ✓
    tensors['attn_lse_intermediates'] = torch.zeros(
        config.num_attention_heads, lse_col_padded,
        device=device, dtype=torch.float32,
    )

    # attn_out_intermediates_t = gl<float, 1, num_attention_heads=32, -1, head_dim=64>
    # → 3D tensor [32, kernel_sm_count, 64] → padded to [1, 32, kernel_sm_count, 64]
    #   b=1 ✓, d=32 ✓
    tensors['attn_out_intermediates'] = torch.zeros(
        config.num_attention_heads, kernel_sm_count, config.head_dim,
        device=device, dtype=torch.float32,
    )

    # ---- MoE 权重 ----
    # moe_weights_t = gl<bf16, -1, -1, -1, hidden_dim>  全动态
    # → 3D tensor [num_layers*num_experts, out_features, hidden_dim]
    # → padded to [1, le, out_features, hidden_dim]
    up_out_features = config.intermediate_size  # 8192
    down_out_features = config.hidden_size      # 2048
    tensors['moe_up_weights'] = torch.randn(
        num_le, up_out_features, config.hidden_size,
        device=device, dtype=dtype,
    ) * 0.01
    tensors['moe_gate_weights'] = torch.randn(
        num_le, up_out_features, config.hidden_size,
        device=device, dtype=dtype,
    ) * 0.01
    # moe_weights_big_t = gl<bf16, -1, -1, -1, intermediate_dim>
    tensors['moe_down_weights'] = torch.randn(
        num_le, down_out_features, config.intermediate_size,
        device=device, dtype=dtype,
    ) * 0.01

    # ---- MoE Routing ----
    # routing_t = gl<int, 1, 1, -1, num_experts_per_tok=2>
    # → 2D tensor [num_layers, 2] → padded to [1, 1, num_layers, 2]
    tensors['moe_expert_indices'] = torch.randint(
        0, config.num_experts,
        (config.num_hidden_layers, config.num_experts_per_tok),
        device=device, dtype=torch.int32,
    )
    tensors['moe_expert_routing_weights'] = torch.softmax(
        torch.randn(config.num_hidden_layers, config.num_experts_per_tok,
                     device=device, dtype=torch.float32),
        dim=-1,
    )

    # ---- MoE 中间缓冲 ----
    # activations_big_indim_t = gl<bf16, 1, 1, 1, intermediate_dim=8192>
    # → 1D tensor [8192]
    tensors['moe_intermediate'] = torch.zeros(
        config.intermediate_size, device=device, dtype=dtype,
    )

    # ---- 融合特有字段 ----
    # attn_done_barrier_t = gl<int, 1, 1, 1, -1>
    # → 1D tensor [batch_size]
    tensors['attn_done_barrier'] = torch.zeros(
        batch_size, device=device, dtype=torch.int32,
    )
    # moe_input_activations_t = gl<bf16, 1, 1, -1, hidden_dim=2048>
    # → 2D tensor [batch_size, hidden_dim]
    tensors['moe_input_activations'] = torch.randn(
        batch_size, config.hidden_size, device=device, dtype=dtype,
    ) * 0.1
    # moe_output_accumulator_t = gl<float, 1, 1, -1, hidden_dim=2048>
    # → 2D tensor [batch_size, hidden_dim]
    tensors['moe_output_accumulator'] = torch.zeros(
        batch_size, config.hidden_size, device=device, dtype=torch.float32,
    )

    return tensors


# ============================================================================
# 指令调度与序列化
# ============================================================================

def create_fused_instructions(config: ModelConfig, seq_len: int, sm_count: int,
                              batch_size: int = 1, layer_idx: int = 0,
                              moe_ready_tokens=None):
    """
    生成融合指令。
    对 layer_idx=0: 只有 attention（没有 MoE overlap）
    对 layer_idx>0: attention + MoE overlap
    """
    from megakernels.demos.fused.instructions import FusedAttnMoE

    instructions = []
    num_kv_heads = config.num_key_value_heads
    gqa_ratio = config.num_attention_heads // num_kv_heads

    # 生成 attention tasks
    attn_tasks = []
    for kv_head_idx in range(num_kv_heads):
        attn_tasks.append({
            'layer_idx': layer_idx,
            'kv_head_idx': kv_head_idx,
            'num_partials': 1,  # decode: 单 partition
            'partial_idx': 0,
        })

    # 生成 MoE tasks
    moe_tasks = []
    if moe_ready_tokens:
        moe_block_size = config.matvec_block_size
        for token_idx in moe_ready_tokens:
            for expert_idx in range(config.num_experts_per_tok):
                # Up projection blocks
                num_up_blocks = config.intermediate_size // moe_block_size
                blocks_per_chunk = max(1, num_up_blocks // max(1, sm_count // 8))
                for start in range(0, num_up_blocks, blocks_per_chunk):
                    end = min(start + blocks_per_chunk, num_up_blocks)
                    moe_tasks.append({
                        'moe_token_idx': token_idx,
                        'moe_expert_idx': expert_idx,
                        'moe_weight_type': 0,
                        'moe_start_block': start,
                        'moe_end_block': end,
                        'moe_reduction_block': 0,
                    })
                    moe_tasks.append({
                        'moe_token_idx': token_idx,
                        'moe_expert_idx': expert_idx,
                        'moe_weight_type': 1,
                        'moe_start_block': start,
                        'moe_end_block': end,
                        'moe_reduction_block': 0,
                    })

                # Down projection blocks
                num_down_blocks = config.hidden_size // moe_block_size
                num_col_splits = config.intermediate_size // config.hidden_size
                for col_idx in range(num_col_splits):
                    blocks_per_chunk_d = max(1, num_down_blocks // max(1, sm_count // 8))
                    for start in range(0, num_down_blocks, blocks_per_chunk_d):
                        end = min(start + blocks_per_chunk_d, num_down_blocks)
                        moe_tasks.append({
                            'moe_token_idx': token_idx,
                            'moe_expert_idx': expert_idx,
                            'moe_weight_type': 2,
                            'moe_start_block': start,
                            'moe_end_block': end,
                            'moe_reduction_block': col_idx,
                        })

    # 配对
    num_attn = len(attn_tasks)
    num_moe = len(moe_tasks)

    for i, attn_task in enumerate(attn_tasks):
        moe_task = moe_tasks[i] if i < num_moe else {
            'moe_token_idx': -1, 'moe_expert_idx': 0,
            'moe_weight_type': 0, 'moe_start_block': 0,
            'moe_end_block': 0, 'moe_reduction_block': 0,
        }
        instructions.append(FusedAttnMoE(**attn_task, **moe_task))

    for i in range(num_attn, num_moe):
        moe_task = moe_tasks[i]
        instructions.append(FusedAttnMoE(
            layer_idx=layer_idx, kv_head_idx=0,
            num_partials=1, partial_idx=0, **moe_task,
        ))

    return instructions


INTS_PER_INSTRUCTION = 32


def serialize_instruction(inst):
    """序列化单条指令为 32 个 int32。"""
    from megakernels.instructions import NoOp
    serialized = inst.serialize()
    padding = INTS_PER_INSTRUCTION - len(serialized)
    assert padding >= 0, f"Instruction too long: {len(serialized)} > {INTS_PER_INSTRUCTION}"
    return serialized + [0] * padding


def tensorize_instructions_for_sms(instructions, sm_count, device):
    """
    Round-robin 分配指令到各 SM，序列化为 tensor。
    返回 (instructions_tensor, timings_tensor)。
    """
    from megakernels.instructions import NoOp

    sm_queues = [[] for _ in range(sm_count)]
    for i, inst in enumerate(instructions):
        sm_queues[i % sm_count].append(inst)

    max_queue_len = max(len(q) for q in sm_queues) if sm_queues else 1
    for q in sm_queues:
        while len(q) < max_queue_len:
            q.append(NoOp())

    flattened = []
    for q in sm_queues:
        for inst in q:
            flattened.append(serialize_instruction(inst))

    instructions_tensor = torch.tensor(
        flattened, dtype=torch.int32, device=device
    ).view(sm_count, max_queue_len, INTS_PER_INSTRUCTION)

    timings_tensor = torch.zeros(
        sm_count, max_queue_len, 128,
        dtype=torch.int32, device=device,
    )

    return instructions_tensor, timings_tensor


# ============================================================================
# PyTorch 参考实现（Baseline）
# ============================================================================

def reference_attention_full(
    q_post_rope: torch.Tensor,  # [hidden_size] = [num_heads * head_dim]
    k_cache: torch.Tensor,      # [num_kv_blocks, num_kv_heads, kv_block_size, head_dim]
    v_cache: torch.Tensor,      # same shape
    config: ModelConfig,
    layer_idx: int,
    seq_len: int,
    num_kv_blocks_total: int,
) -> torch.Tensor:
    """
    PyTorch 参考 attention 实现（全部 heads）。
    返回 attn_out: [hidden_size]
    """
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    gqa_ratio = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    Q = q_post_rope.view(num_heads, head_dim).float()
    attn_out = torch.zeros(num_heads, head_dim, device=Q.device, dtype=torch.float32)

    # 每个 layer 的 KV cache 偏移
    kv_offset = layer_idx * num_kv_blocks_total

    for kv_h in range(num_kv_heads):
        q_start = kv_h * gqa_ratio
        Q_group = Q[q_start:q_start + gqa_ratio]  # [gqa_ratio, head_dim]

        # 展平 KV cache for this head
        K_all = k_cache[kv_offset:kv_offset + num_kv_blocks_total, kv_h].reshape(-1, head_dim)[:seq_len].float()
        V_all = v_cache[kv_offset:kv_offset + num_kv_blocks_total, kv_h].reshape(-1, head_dim)[:seq_len].float()

        scores = torch.matmul(Q_group, K_all.T) * scale
        weights = F.softmax(scores, dim=-1)
        O_group = torch.matmul(weights, V_all)

        attn_out[q_start:q_start + gqa_ratio] = O_group

    return attn_out.to(torch.bfloat16).view(-1)


def reference_moe_forward(
    input_act: torch.Tensor,    # [hidden_size]
    moe_up: torch.Tensor,       # [le, intermediate, hidden]
    moe_gate: torch.Tensor,     # same
    moe_down: torch.Tensor,     # [le, hidden, intermediate]
    expert_indices: torch.Tensor,  # [num_experts_per_tok]
    expert_weights: torch.Tensor,  # [num_experts_per_tok]
    config: ModelConfig,
    layer_idx: int,
) -> torch.Tensor:
    """
    PyTorch 参考 MoE 前向。
    简化实现：直接用矩阵乘法。
    """
    output = torch.zeros_like(input_act, dtype=torch.float32)
    x = input_act.float()

    num_experts_offset = layer_idx * config.num_experts

    for i in range(config.num_experts_per_tok):
        expert_id = expert_indices[i].item()
        weight = expert_weights[i].item()

        # 重构 up/gate 权重: [intermediate, hidden]
        up_w = moe_up[num_experts_offset + expert_id].reshape(-1, config.hidden_size).float()
        gate_w = moe_gate[num_experts_offset + expert_id].reshape(-1, config.hidden_size).float()
        down_w = moe_down[num_experts_offset + expert_id].reshape(-1, config.intermediate_size).float()

        up_out = torch.matmul(up_w, x)
        gate_out = torch.matmul(gate_w, x)
        gate_out = gate_out * torch.sigmoid(gate_out)  # SiLU
        intermediate = up_out * gate_out
        down_out = torch.matmul(down_w, intermediate)

        output += down_out * weight

    return output.to(torch.bfloat16)


# ============================================================================
# Baseline：串行执行 Attention + MoE
# ============================================================================

def baseline_sequential_forward(
    tensors: dict,
    config: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
):
    """
    串行执行: 先跑 attention（所有 heads），再跑 MoE。
    用 PyTorch 实现，模拟的是两个 kernel 的开销。
    这里只用做时间对比参考。
    """
    num_kv_blocks = math.ceil(seq_len / config.kv_block_size)

    # Attention
    attn_out = reference_attention_full(
        tensors['q_post_rope'],
        tensors['k_cache'],
        tensors['v_cache'],
        config, 0, seq_len, num_kv_blocks,
    )

    # MoE
    moe_out = reference_moe_forward(
        tensors['moe_input_activations'][0],
        tensors['moe_up_weights'],
        tensors['moe_gate_weights'],
        tensors['moe_down_weights'],
        tensors['moe_expert_indices'][0],
        tensors['moe_expert_routing_weights'][0],
        config, 0,
    )

    return attn_out, moe_out


# ============================================================================
# CUDA Kernel 调用
# ============================================================================

def load_fused_kernel(mk_dir: Path):
    """加载编译好的 fused megakernel .so 文件。"""
    sys.path.insert(0, str(mk_dir.expanduser().absolute()))
    try:
        from mk_fused_attn_moe import mk_fused_attn_moe
        return mk_fused_attn_moe
    except ImportError as e:
        print(f"❌ 无法加载编译好的 kernel: {e}")
        print(f"   请确保已在 {mk_dir} 下运行 make GPU=H20 PYTHON_VERSION=3.11")
        sys.exit(1)


def run_fused_kernel(mk_func, tensors, instructions_tensor, timings_tensor,
                     config, seq_len, batch_size):
    """
    调用编译好的 fused CUDA kernel。
    参数顺序必须与 main.cu 中的 pybind11 binding 完全一致。
    """
    # 【关键】预填充 barriers：
    # attention consumer 会 spin-wait 在 Bar[layer_idx, 0, head_idx] >= 4，
    # 这是等待前序 RMS+QKV matmul kernel 完成的信号。
    # 在独立 benchmark 中没有前序 kernel，必须预填充。
    tensors['barriers'][:, 0, :config.num_attention_heads] = 4

    mk_func(
        # VM 基础设施
        tensors['barriers'],
        instructions_tensor,
        timings_tensor,
        # 模型权重
        tensors['qkv_weights'],
        tensors['attn_norm_weights'],
        tensors['o_weights'],
        tensors['mlp_norm_weights'],
        # KV cache (已经是 4D)
        tensors['k_cache'],
        tensors['v_cache'],
        # RoPE
        tensors['rope_cos'],
        tensors['rope_sin'],
        # Activation 缓冲区
        tensors['hidden_states'],
        tensors['q_post_rope'],
        tensors['attn_out'],
        tensors['attn_lse_intermediates'],
        tensors['attn_out_intermediates'],
        # MoE 权重
        tensors['moe_up_weights'],
        tensors['moe_gate_weights'],
        tensors['moe_down_weights'],
        # MoE Routing
        tensors['moe_expert_indices'],
        tensors['moe_expert_routing_weights'],
        # MoE 中间缓冲
        tensors['moe_intermediate'],
        # 融合特有字段
        tensors['attn_done_barrier'],
        tensors['moe_input_activations'],
        tensors['moe_output_accumulator'],
        # 标量
        seq_len - 1,        # pos_id
        1.0 / math.sqrt(config.head_dim),  # attn_scale
        config.rms_norm_eps,  # rms_norm_eps
        batch_size,          # batch_size
        0,                   # current_token_idx
        -1,                  # moe_token_idx
        True,                # skip_attn_reduction
        stream=torch.cuda.current_stream(),
    )


# ============================================================================
# 性能测试
# ============================================================================

def benchmark_kernel(fn, warmup=20, iters=100, label="kernel"):
    """
    通用 CUDA kernel benchmark 工具。
    使用 CUDA events 精确计时。
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 精确计时
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_us = [t * 1000 for t in times_ms]

    avg_us = mean(times_us)
    std_us = stdev(times_us) if len(times_us) > 1 else 0.0
    min_us = min(times_us)
    max_us = max(times_us)
    p50_us = sorted(times_us)[len(times_us) // 2]
    p99_us = sorted(times_us)[int(len(times_us) * 0.99)]

    return {
        'label': label,
        'avg_us': avg_us,
        'std_us': std_us,
        'min_us': min_us,
        'max_us': max_us,
        'p50_us': p50_us,
        'p99_us': p99_us,
        'times_us': times_us,
    }


def print_benchmark_result(result, batch_size=1):
    """打印 benchmark 结果。"""
    avg = result['avg_us']
    tokens_per_sec = batch_size / (avg / 1e6) if avg > 0 else 0

    print(f"  {result['label']}:")
    print(f"    Average:   {avg:10.1f} µs  ({avg/1000:.3f} ms)")
    print(f"    Std dev:   {result['std_us']:10.1f} µs")
    print(f"    Min:       {result['min_us']:10.1f} µs")
    print(f"    P50:       {result['p50_us']:10.1f} µs")
    print(f"    P99:       {result['p99_us']:10.1f} µs")
    print(f"    Max:       {result['max_us']:10.1f} µs")
    print(f"    Tokens/s:  {tokens_per_sec:,.0f}")
    print()


# ============================================================================
# 正确性测试
# ============================================================================

def test_correctness(mk_func, tensors, config, seq_len, sm_count, device):
    """
    验证 CUDA kernel 的 attention 输出是否与 PyTorch 参考一致。
    """
    print("=" * 70)
    print("  正确性验证: CUDA Kernel vs PyTorch Reference")
    print("=" * 70)

    batch_size = 1
    num_kv_blocks = math.ceil(seq_len / config.kv_block_size)

    # ---- 只测试 attention（layer 0, 无 MoE overlap）----
    instructions = create_fused_instructions(
        config, seq_len, sm_count, batch_size=batch_size,
        layer_idx=0, moe_ready_tokens=None,
    )
    inst_tensor, timing_tensor = tensorize_instructions_for_sms(
        instructions, sm_count, device,
    )

    # 清空输出 buffer
    tensors['attn_out'].zero_()
    tensors['attn_out_intermediates'].zero_()
    tensors['attn_lse_intermediates'].zero_()
    tensors['barriers'].zero_()

    # 运行 CUDA kernel
    run_fused_kernel(mk_func, tensors, inst_tensor, timing_tensor,
                     config, seq_len, batch_size)
    torch.cuda.synchronize()

    # PyTorch 参考
    ref_attn_out = reference_attention_full(
        tensors['q_post_rope'], tensors['k_cache'], tensors['v_cache'],
        config, 0, seq_len, num_kv_blocks,
    )

    # 对比
    cuda_out = tensors['attn_out'].float()
    ref_out = ref_attn_out.float()

    abs_diff = (cuda_out - ref_out).abs()
    rel_diff = 2 * abs_diff / (cuda_out.abs() + ref_out.abs() + 1e-8)

    max_abs = abs_diff.max().item()
    mean_rel = rel_diff.mean().item()

    print(f"\n  Attention 输出对比:")
    print(f"    Max absolute diff:  {max_abs:.6e}")
    print(f"    Mean relative diff: {mean_rel:.6e}")

    # 非零输出检查
    cuda_norm = cuda_out.norm().item()
    ref_norm = ref_out.norm().item()
    print(f"    CUDA output norm:   {cuda_norm:.4f}")
    print(f"    Ref output norm:    {ref_norm:.4f}")

    # 逐 head 对比
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    print(f"\n  逐 Head 对比 (前 4 个 heads):")
    for h in range(min(4, num_heads)):
        start = h * head_dim
        end = start + head_dim
        hd_abs = abs_diff[start:end].max().item()
        hd_cos = F.cosine_similarity(
            cuda_out[start:end].unsqueeze(0),
            ref_out[start:end].unsqueeze(0),
        ).item()
        status = "✓" if hd_abs < 0.1 else "✗"
        print(f"    Head {h:3d}: max_abs_diff={hd_abs:.4e}, "
              f"cosine_sim={hd_cos:.6f} {status}")

    passed = max_abs < 0.5  # bf16 tolerance
    print(f"\n  {'✓ 正确性测试通过' if passed else '✗ 正确性测试失败'} "
          f"(threshold=0.5, actual={max_abs:.4e})")
    print()
    return passed


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="融合 Attention + MoE Megakernel 端到端性能测试"
    )
    parser.add_argument("--seq-len", type=int, default=128,
                        help="KV cache 序列长度")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (decode 阶段通常为 1)")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="测试的 layer 数量")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup 迭代次数")
    parser.add_argument("--iters", type=int, default=100,
                        help="计时迭代次数")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mk-dir", type=str, default=None,
                        help="编译好的 .so 文件目录 (默认: demos/fused-attn-moe/)")
    parser.add_argument("--skip-correctness", action="store_true",
                        help="跳过正确性验证")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过 baseline 对比")
    parser.add_argument("--sweep-seq-lens", action="store_true",
                        help="扫描多个 seq_len")
    args = parser.parse_args()

    # ---- 设备信息 ----
    device = args.device
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    hw_sm_count = props.multi_processor_count

    # kernel 编译时的 SM count — get_worker_id() = smid，必须匹配硬件
    # COMPILED_SM_COUNT=0 表示自动使用硬件值
    kernel_sm_count = COMPILED_SM_COUNT if COMPILED_SM_COUNT > 0 else hw_sm_count

    print("=" * 70)
    print("  融合 Attention + MoE Megakernel — 端到端 CUDA 性能测试")
    print("=" * 70)
    print(f"  GPU:           {props.name}")
    print(f"  HW SM Count:   {hw_sm_count}")
    print(f"  Kernel SMs:    {kernel_sm_count} (compiled)")
    print(f"  VRAM:          {props.total_memory // 1024**3} GB")
    print(f"  CUDA:          {torch.version.cuda}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  Seq Length:    {args.seq_len}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Layers:        {args.num_layers}")
    print(f"  Warmup:        {args.warmup}")
    print(f"  Iters:         {args.iters}")
    print()

    # ---- 加载 kernel ----
    if args.mk_dir:
        mk_dir = Path(args.mk_dir)
    else:
        mk_dir = Path(__file__).parent.parent.parent.parent / "demos" / "fused-attn-moe"

    mk_func = load_fused_kernel(mk_dir)
    print(f"  ✓ 成功加载 kernel: {mk_dir}")
    print()

    # ---- 模型配置 ----
    config = ModelConfig(num_hidden_layers=args.num_layers)

    # ---- Seq len 扫描模式 ----
    if args.sweep_seq_lens:
        seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
    else:
        seq_lens = [args.seq_len]

    for seq_len in seq_lens:
        print("=" * 70)
        print(f"  序列长度: {seq_len}")
        print("=" * 70)

        # ---- 创建 tensor ----
        tensors = create_model_tensors(config, seq_len, kernel_sm_count, args.batch_size, device)

        # ---- 正确性验证 ----
        if not args.skip_correctness:
            test_correctness(mk_func, tensors, config, seq_len, kernel_sm_count, device)

        # ---- 生成指令 (Attention only, layer 0) ----
        instructions_attn_only = create_fused_instructions(
            config, seq_len, kernel_sm_count, args.batch_size,
            layer_idx=0, moe_ready_tokens=None,
        )
        inst_tensor_attn, timing_tensor_attn = tensorize_instructions_for_sms(
            instructions_attn_only, kernel_sm_count, device,
        )

        # ---- 生成指令 (Fused Attention + MoE, layer 1) ----
        moe_ready = list(range(args.batch_size))
        instructions_fused = create_fused_instructions(
            config, seq_len, kernel_sm_count, args.batch_size,
            layer_idx=0, moe_ready_tokens=moe_ready,
        )
        inst_tensor_fused, timing_tensor_fused = tensorize_instructions_for_sms(
            instructions_fused, kernel_sm_count, device,
        )

        print(f"  指令统计:")
        print(f"    Attention-only: {len(instructions_attn_only)} 条指令")
        print(f"    Fused (Attn+MoE): {len(instructions_fused)} 条指令")
        attn_only_count = sum(1 for i in instructions_fused if i.moe_token_idx < 0)
        fused_count = sum(1 for i in instructions_fused if i.moe_token_idx >= 0)
        print(f"      其中: {attn_only_count} attn-only, {fused_count} fused")
        print()

        # ======== Benchmark 1: Attention-only Kernel ========
        print("-" * 70)
        print("  Benchmark 1: Attention-only (Fused Kernel, 无 MoE overlap)")
        print("-" * 70)

        def run_attn_only():
            tensors['barriers'].zero_()
            tensors['attn_out'].zero_()
            run_fused_kernel(mk_func, tensors, inst_tensor_attn, timing_tensor_attn,
                             config, seq_len, args.batch_size)

        result_attn_only = benchmark_kernel(
            run_attn_only, warmup=args.warmup, iters=args.iters,
            label="Fused Kernel (Attention-only)"
        )
        print_benchmark_result(result_attn_only, args.batch_size)

        # ======== Benchmark 2: Fused Attention + MoE Kernel ========
        print("-" * 70)
        print("  Benchmark 2: Fused Kernel (Attention + MoE overlap)")
        print("-" * 70)

        def run_fused():
            tensors['barriers'].zero_()
            tensors['attn_out'].zero_()
            tensors['moe_output_accumulator'].zero_()
            tensors['moe_intermediate'].zero_()
            tensors['attn_done_barrier'].zero_()
            run_fused_kernel(mk_func, tensors, inst_tensor_fused, timing_tensor_fused,
                             config, seq_len, args.batch_size)

        result_fused = benchmark_kernel(
            run_fused, warmup=args.warmup, iters=args.iters,
            label="Fused Kernel (Attn + MoE)"
        )
        print_benchmark_result(result_fused, args.batch_size)

        # ======== Benchmark 3: Baseline (Sequential PyTorch) ========
        if not args.skip_baseline:
            print("-" * 70)
            print("  Benchmark 3: Baseline (PyTorch Sequential Attn + MoE)")
            print("-" * 70)

            def run_baseline():
                baseline_sequential_forward(tensors, config, seq_len, args.batch_size)

            result_baseline = benchmark_kernel(
                run_baseline, warmup=args.warmup, iters=args.iters,
                label="PyTorch Sequential (Attn + MoE)"
            )
            print_benchmark_result(result_baseline, args.batch_size)

        # ======== 汇总 ========
        print("=" * 70)
        print("  性能汇总")
        print("=" * 70)

        attn_us = result_attn_only['avg_us']
        fused_us = result_fused['avg_us']

        print(f"  Attention-only kernel:  {attn_us:10.1f} µs")
        print(f"  Fused (Attn+MoE):       {fused_us:10.1f} µs")

        # 计算 MoE 开销
        moe_overhead = fused_us - attn_us
        print(f"  MoE 额外开销:           {moe_overhead:10.1f} µs "
              f"({moe_overhead/fused_us*100:.1f}%)")

        if not args.skip_baseline:
            baseline_us = result_baseline['avg_us']
            speedup = baseline_us / fused_us if fused_us > 0 else 0

            print(f"  Baseline (Sequential):  {baseline_us:10.1f} µs")
            print(f"  加速比 (Baseline/Fused): {speedup:.2f}×")
            print()

            # Tokens/s 对比
            fused_tps = args.batch_size / (fused_us / 1e6)
            baseline_tps = args.batch_size / (baseline_us / 1e6)
            print(f"  Tokens/s:")
            print(f"    Fused kernel:     {fused_tps:>12,.0f} tok/s")
            print(f"    Baseline:         {baseline_tps:>12,.0f} tok/s")
        else:
            fused_tps = args.batch_size / (fused_us / 1e6)
            print(f"  Tokens/s (Fused):   {fused_tps:>12,.0f} tok/s")

        # 理论分析
        print()
        print(f"  理论分析 (seq_len={seq_len}):")
        kv_bytes = seq_len * config.head_dim * 2 * 2 * config.num_key_value_heads  # K+V, bf16
        moe_weight_bytes = (
            2 * config.intermediate_size * config.hidden_size * 2 +  # up+gate
            config.hidden_size * config.intermediate_size * 2        # down
        ) * config.num_experts_per_tok  # bf16
        print(f"    KV cache 数据量:    {kv_bytes / 1024:.1f} KB")
        print(f"    MoE 权重数据量:     {moe_weight_bytes / 1024 / 1024:.1f} MB")
        print(f"    MoE/KV 比值:        {moe_weight_bytes / kv_bytes:.1f}×")
        print()


if __name__ == "__main__":
    main()
