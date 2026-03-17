#!/usr/bin/env python3
"""
============================================================================
  独立 Benchmark 脚本 — 无需安装 megakernels 包
============================================================================

直接在 demos/fused-attn-moe/ 目录运行:
  cd demos/fused-attn-moe
  python bench_standalone.py

或指定参数:
  python bench_standalone.py --seq-len 512 --warmup 50 --iters 200 --sweep
"""

import argparse
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F


# ============================================================================
# 模型配置（与编译时宏一致）
# ============================================================================

HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 8192
HEAD_DIM = 64
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
KV_BLOCK_SIZE = 16
MATVEC_BLOCK_SIZE = 16
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
NUM_LAYERS = 16  # FUSED_NUM_LAYERS in main.cu
RMS_NORM_EPS = 1e-5
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS
# 编译时写死的 SM count（main.cu 中通过 FUSED_SM_COUNT 宏或默认值决定）
# kernel 的 grid() 返回 dim3(sm_count)，指令/timing tensor 必须用编译时 SM count
# get_worker_id() 返回 smid（硬件 SM ID），所以必须与实际硬件 SM 数匹配！
# 如果不确定编译时用的值，设为 0 表示运行时自动用硬件 SM count
COMPILED_SM_COUNT = 0  # 0 = 自动检测（使用硬件 SM 数量）

INTS_PER_INSTRUCTION = 32
OPCODE_FUSED = 9


# ============================================================================
# 加载编译好的 Kernel
# ============================================================================

def load_kernel():
    """加载当前目录下的 mk_fused_attn_moe.so"""
    sys.path.insert(0, str(Path(__file__).parent.absolute()))
    try:
        from mk_fused_attn_moe import mk_fused_attn_moe
        return mk_fused_attn_moe
    except ImportError as e:
        print(f"❌ 无法加载 kernel: {e}")
        print(f"   请先运行: make GPU=H20 PYTHON_VERSION=3.11")
        sys.exit(1)


# ============================================================================
# 创建 GPU Tensors
# ============================================================================

def create_tensors(seq_len, kernel_sm_count, batch_size=1, device="cuda:0"):
    """
    创建所有必需的 GPU tensors。
    形状必须与 fused_globals_t 中 gl<> 类型完全匹配。

    pyutils.cuh 的 from_object<GL> 会将 tensor 左填充 1 到 4D 后检查：
      - gl<T, B, D, R, C> 中 B/D/R/C > 0 的维度必须精确匹配
      - -1 表示动态维度，不检查

    【关键规则】
      - gl<T, 1, ...> 的 batch=1 → 用 3D 或更低维 tensor（自动填充 batch=1）
      - 如果传 4D tensor，第一维就是 batch，必须为 1
      - gl<T, -1, -1, -1, C> 全动态 → 4D tensor 即可，不检查 batch
    """
    dtype = torch.bfloat16
    num_kv_blocks = math.ceil(seq_len / KV_BLOCK_SIZE)
    total_heads = NUM_ATTENTION_HEADS + NUM_KV_HEADS * 2
    num_le = NUM_LAYERS * NUM_EXPERTS

    # 编译时 SM count 用于 LSE intermediates 的 column padding
    lse_col_padded = ((kernel_sm_count + 15) // 16) * 16

    t = {}

    # ---- VM 基础设施 ----
    # barriers: gl<uint, 1, -1, -1, N>  where N = num_attention_heads + 2*num_kv_heads = 48
    # → 3D tensor [NUM_LAYERS, 12, 48] → padded to [1, NUM_LAYERS, 12, 48]  b=1 ✓
    t['barriers'] = torch.zeros(NUM_LAYERS, 12, total_heads,
                                dtype=torch.int32, device=device)

    # ---- 模型权重 ----
    # weights_t = gl<bf16, 1, -1, -1, hidden_dim=2048>
    # 原始模型中: stacked_qkv = [num_layers, total_heads*head_dim, hidden_dim]  3D
    # → padded to [1, num_layers, total_heads*head_dim, hidden_dim]  b=1 ✓
    qkv_out_features = total_heads * HEAD_DIM  # (32+8+8)*64 = 3072
    t['qkv_weights'] = torch.randn(NUM_LAYERS, qkv_out_features, HIDDEN_DIM,
                                   device=device, dtype=dtype) * 0.01

    # norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim>
    # → 2D tensor [num_layers, hidden_dim] → padded to [1, 1, num_layers, hidden_dim]
    #   b=1 ✓, d=1 ✓
    t['attn_norm_weights'] = torch.ones(NUM_LAYERS, HIDDEN_DIM,
                                        device=device, dtype=dtype)

    # o_weights: same as weights_t = gl<bf16, 1, -1, -1, hidden_dim>
    o_out_features = NUM_ATTENTION_HEADS * HEAD_DIM  # 32*64 = 2048
    t['o_weights'] = torch.randn(NUM_LAYERS, o_out_features, HIDDEN_DIM,
                                 device=device, dtype=dtype) * 0.01

    t['mlp_norm_weights'] = torch.ones(NUM_LAYERS, HIDDEN_DIM,
                                       device=device, dtype=dtype)

    # ---- KV cache ----
    # kv_cache_t = gl<bf16, -1, -1, -1, head_dim=64>  全动态
    # → 4D tensor [num_layers*num_kv_blocks, num_kv_heads, kv_block_size, head_dim]
    t['k_cache'] = torch.randn(NUM_LAYERS * num_kv_blocks, NUM_KV_HEADS,
                                KV_BLOCK_SIZE, HEAD_DIM, device=device, dtype=dtype) * 0.1
    t['v_cache'] = torch.randn(NUM_LAYERS * num_kv_blocks, NUM_KV_HEADS,
                                KV_BLOCK_SIZE, HEAD_DIM, device=device, dtype=dtype) * 0.1

    # ---- RoPE ----
    # rope_table_t = gl<float, 1, 1, -1, head_dim=64>
    # → 2D tensor [seq_len, head_dim] → padded to [1, 1, seq_len, head_dim]
    t['rope_cos'] = torch.ones(seq_len + 16, HEAD_DIM, device=device, dtype=torch.float32)
    t['rope_sin'] = torch.zeros(seq_len + 16, HEAD_DIM, device=device, dtype=torch.float32)

    # ---- Activations ----
    # activations_t = gl<bf16, 1, 1, 1, hidden_dim=2048>  所有维度固定
    # → 1D tensor [hidden_dim] → padded to [1, 1, 1, hidden_dim]
    t['hidden_states'] = torch.randn(HIDDEN_DIM, device=device, dtype=dtype) * 0.1
    t['q_post_rope'] = torch.randn(HIDDEN_DIM, device=device, dtype=dtype) * 0.1
    t['attn_out'] = torch.zeros(HIDDEN_DIM, device=device, dtype=dtype)

    # attn_lse_intermediates_t = gl<float, 1, 1, num_attention_heads=32, -1>
    # → 2D tensor [32, lse_col_padded] → padded to [1, 1, 32, lse_col_padded]
    #   b=1 ✓, d=1 ✓, r=32 ✓
    t['attn_lse_intermediates'] = torch.zeros(
        NUM_ATTENTION_HEADS, lse_col_padded, device=device, dtype=torch.float32)

    # attn_out_intermediates_t = gl<float, 1, num_attention_heads=32, -1, head_dim=64>
    # → 3D tensor [32, kernel_sm_count, 64] → padded to [1, 32, kernel_sm_count, 64]
    #   b=1 ✓, d=32 ✓
    t['attn_out_intermediates'] = torch.zeros(
        NUM_ATTENTION_HEADS, kernel_sm_count, HEAD_DIM,
        device=device, dtype=torch.float32)

    # ---- MoE 权重 ----
    # moe_weights_t = gl<bf16, -1, -1, -1, hidden_dim=2048>  全动态
    # 原始: [num_layers*num_experts, intermediate/block_size, hidden_dim]  3D
    # → padded to [1, le, blocks, hidden_dim] — 但由于全动态，4D 也可以
    # 为安全起见用 3D: [le, out_features, hidden_dim]
    up_out_features = INTERMEDIATE_DIM    # 8192
    down_out_features = HIDDEN_DIM        # 2048
    t['moe_up_weights'] = torch.randn(num_le, up_out_features, HIDDEN_DIM,
                                      device=device, dtype=dtype) * 0.01
    t['moe_gate_weights'] = torch.randn(num_le, up_out_features, HIDDEN_DIM,
                                        device=device, dtype=dtype) * 0.01
    # moe_weights_big_t = gl<bf16, -1, -1, -1, intermediate_dim=8192>
    t['moe_down_weights'] = torch.randn(num_le, down_out_features, INTERMEDIATE_DIM,
                                        device=device, dtype=dtype) * 0.01

    # ---- MoE Routing ----
    # routing_t = gl<int, 1, 1, -1, num_experts_per_tok=2>
    # → 2D tensor [NUM_LAYERS, 2] → padded to [1, 1, NUM_LAYERS, 2]
    t['moe_expert_indices'] = torch.randint(0, NUM_EXPERTS,
        (NUM_LAYERS, NUM_EXPERTS_PER_TOK), device=device, dtype=torch.int32)
    # routing_weight_t = gl<float, 1, 1, -1, 2>
    t['moe_expert_routing_weights'] = torch.softmax(
        torch.randn(NUM_LAYERS, NUM_EXPERTS_PER_TOK, device=device, dtype=torch.float32),
        dim=-1)

    # ---- MoE 中间缓冲 ----
    # activations_big_indim_t = gl<bf16, 1, 1, 1, intermediate_dim=8192>
    # → 1D tensor [8192]
    t['moe_intermediate'] = torch.zeros(INTERMEDIATE_DIM, device=device, dtype=dtype)

    # ---- 融合特有字段 ----
    # attn_done_barrier_t = gl<int, 1, 1, 1, -1>
    # → 1D tensor [batch_size] → padded to [1, 1, 1, batch_size]
    t['attn_done_barrier'] = torch.zeros(batch_size, device=device, dtype=torch.int32)
    # moe_input_activations_t = gl<bf16, 1, 1, -1, hidden_dim=2048>
    # → 2D tensor [batch_size, hidden_dim] → padded to [1, 1, batch_size, hidden_dim]
    t['moe_input_activations'] = torch.randn(batch_size, HIDDEN_DIM,
                                             device=device, dtype=dtype) * 0.1
    # moe_output_accumulator_t = gl<float, 1, 1, -1, hidden_dim=2048>
    # → 2D tensor [batch_size, hidden_dim]
    t['moe_output_accumulator'] = torch.zeros(batch_size, HIDDEN_DIM,
                                              device=device, dtype=torch.float32)

    return t


# ============================================================================
# 指令生成与序列化
# ============================================================================

def make_fused_instruction(layer_idx, kv_head_idx, num_partials, partial_idx,
                           moe_token_idx=-1, moe_expert_idx=0,
                           moe_weight_type=0, moe_start_block=0,
                           moe_end_block=0, moe_reduction_block=0):
    """生成一条融合指令的 int32 序列。"""
    words = [
        OPCODE_FUSED,
        layer_idx,
        kv_head_idx,
        num_partials,
        partial_idx,
        moe_token_idx,
        moe_expert_idx,
        moe_weight_type,
        moe_start_block,
        moe_end_block,
        moe_reduction_block,
    ]
    return words + [0] * (INTS_PER_INSTRUCTION - len(words))


def make_noop():
    return [0] * INTS_PER_INSTRUCTION


def create_attn_only_instructions(sm_count, layer_idx=0):
    """生成 attention-only 指令。"""
    instructions = []
    for kv_head in range(NUM_KV_HEADS):
        instructions.append(make_fused_instruction(
            layer_idx=layer_idx, kv_head_idx=kv_head,
            num_partials=1, partial_idx=0,
        ))
    return instructions


def create_fused_instructions(sm_count, layer_idx=0, batch_size=1):
    """生成融合 Attention + MoE 指令。"""
    instructions = []

    # Attention tasks
    attn_tasks = []
    for kv_head in range(NUM_KV_HEADS):
        attn_tasks.append({
            'layer_idx': layer_idx, 'kv_head_idx': kv_head,
            'num_partials': 1, 'partial_idx': 0,
        })

    # MoE tasks
    moe_tasks = []
    for token_idx in range(batch_size):
        for expert_idx in range(NUM_EXPERTS_PER_TOK):
            # Up + Gate
            up_blocks = INTERMEDIATE_DIM // MATVEC_BLOCK_SIZE
            chunks = max(1, up_blocks // max(1, sm_count // 8))
            for start in range(0, up_blocks, chunks):
                end = min(start + chunks, up_blocks)
                moe_tasks.append({
                    'moe_token_idx': token_idx, 'moe_expert_idx': expert_idx,
                    'moe_weight_type': 0, 'moe_start_block': start,
                    'moe_end_block': end, 'moe_reduction_block': 0,
                })
                moe_tasks.append({
                    'moe_token_idx': token_idx, 'moe_expert_idx': expert_idx,
                    'moe_weight_type': 1, 'moe_start_block': start,
                    'moe_end_block': end, 'moe_reduction_block': 0,
                })
            # Down
            down_blocks = HIDDEN_DIM // MATVEC_BLOCK_SIZE
            col_splits = INTERMEDIATE_DIM // HIDDEN_DIM
            for col in range(col_splits):
                chunks_d = max(1, down_blocks // max(1, sm_count // 8))
                for start in range(0, down_blocks, chunks_d):
                    end = min(start + chunks_d, down_blocks)
                    moe_tasks.append({
                        'moe_token_idx': token_idx, 'moe_expert_idx': expert_idx,
                        'moe_weight_type': 2, 'moe_start_block': start,
                        'moe_end_block': end, 'moe_reduction_block': col,
                    })

    # Pair attn + moe tasks
    for i, at in enumerate(attn_tasks):
        mt = moe_tasks[i] if i < len(moe_tasks) else {
            'moe_token_idx': -1, 'moe_expert_idx': 0,
            'moe_weight_type': 0, 'moe_start_block': 0,
            'moe_end_block': 0, 'moe_reduction_block': 0,
        }
        instructions.append(make_fused_instruction(**at, **mt))

    for i in range(len(attn_tasks), len(moe_tasks)):
        mt = moe_tasks[i]
        instructions.append(make_fused_instruction(
            layer_idx=layer_idx, kv_head_idx=0,
            num_partials=1, partial_idx=0, **mt,
        ))

    return instructions


def tensorize(instructions, sm_count, device):
    """Round-robin 分配指令到 SM 并序列化为 tensor。"""
    queues = [[] for _ in range(sm_count)]
    for i, inst in enumerate(instructions):
        queues[i % sm_count].append(inst)

    max_len = max(len(q) for q in queues) if queues else 1
    for q in queues:
        while len(q) < max_len:
            q.append(make_noop())

    flat = []
    for q in queues:
        flat.extend(q)

    inst_t = torch.tensor(flat, dtype=torch.int32, device=device).view(
        sm_count, max_len, INTS_PER_INSTRUCTION)
    timing_t = torch.zeros(sm_count, max_len, 128, dtype=torch.int32, device=device)
    return inst_t, timing_t


# ============================================================================
# Kernel 调用
# ============================================================================

def call_kernel(mk_func, t, inst_t, timing_t, seq_len, batch_size):
    """调用 CUDA kernel，参数顺序与 main.cu pybind11 一致。"""
    mk_func(
        t['barriers'], inst_t, timing_t,
        t['qkv_weights'], t['attn_norm_weights'], t['o_weights'], t['mlp_norm_weights'],
        t['k_cache'], t['v_cache'],
        t['rope_cos'], t['rope_sin'],
        t['hidden_states'], t['q_post_rope'], t['attn_out'],
        t['attn_lse_intermediates'], t['attn_out_intermediates'],
        t['moe_up_weights'], t['moe_gate_weights'], t['moe_down_weights'],
        t['moe_expert_indices'], t['moe_expert_routing_weights'],
        t['moe_intermediate'],
        t['attn_done_barrier'], t['moe_input_activations'], t['moe_output_accumulator'],
        seq_len - 1,                       # pos_id
        1.0 / math.sqrt(HEAD_DIM),         # attn_scale
        RMS_NORM_EPS,                       # rms_norm_eps
        batch_size,                         # batch_size
        0,                                  # current_token_idx
        -1,                                 # moe_token_idx
        True,                               # skip_attn_reduction
        stream=torch.cuda.current_stream(),
    )


# ============================================================================
# Benchmark 工具
# ============================================================================

def benchmark(fn, warmup=20, iters=100):
    """CUDA events 精确计时。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)]  # µs
    return {
        'avg': mean(times),
        'std': stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'p50': sorted(times)[len(times) // 2],
        'p99': sorted(times)[int(len(times) * 0.99)],
    }


def fmt(us):
    if us >= 1000:
        return f"{us/1000:.2f} ms"
    return f"{us:.1f} µs"


# ============================================================================
# PyTorch Baseline
# ============================================================================

def pytorch_attention(q, k_cache, v_cache, seq_len, layer_idx, num_kv_blocks):
    """PyTorch 参考 attention。"""
    scale = 1.0 / math.sqrt(HEAD_DIM)
    Q = q.view(NUM_ATTENTION_HEADS, HEAD_DIM).float()
    out = torch.zeros_like(Q)
    offset = layer_idx * num_kv_blocks

    for kv_h in range(NUM_KV_HEADS):
        qs = kv_h * GQA_RATIO
        Q_g = Q[qs:qs + GQA_RATIO]
        K = k_cache[offset:offset + num_kv_blocks, kv_h].reshape(-1, HEAD_DIM)[:seq_len].float()
        V = v_cache[offset:offset + num_kv_blocks, kv_h].reshape(-1, HEAD_DIM)[:seq_len].float()
        scores = Q_g @ K.T * scale
        w = F.softmax(scores, dim=-1)
        out[qs:qs + GQA_RATIO] = w @ V

    return out.bfloat16().view(-1)


def pytorch_moe(input_act, up_w, gate_w, down_w, expert_ids, expert_ws, layer_idx):
    """PyTorch 参考 MoE。"""
    x = input_act.float()
    out = torch.zeros_like(x)
    offset = layer_idx * NUM_EXPERTS

    for i in range(NUM_EXPERTS_PER_TOK):
        eid = expert_ids[i].item()
        w = expert_ws[i].item()
        up = up_w[offset + eid].reshape(-1, HIDDEN_DIM).float() @ x
        gate = gate_w[offset + eid].reshape(-1, HIDDEN_DIM).float() @ x
        gate = gate * torch.sigmoid(gate)
        inter = up * gate
        down = down_w[offset + eid].reshape(-1, INTERMEDIATE_DIM).float() @ inter
        out += down * w

    return out.bfloat16()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fused Attn+MoE Standalone Benchmark")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sweep", action="store_true", help="扫描多个 seq_len")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    device = args.device
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    hw_sm_count = props.multi_processor_count

    # kernel 编译时的 SM count — grid 大小 = sm_count
    # get_worker_id() 返回 smid（硬件 SM ID），所以编译时 sm_count 必须 == 硬件 SM 数
    # COMPILED_SM_COUNT=0 表示自动使用硬件值
    kernel_sm_count = COMPILED_SM_COUNT if COMPILED_SM_COUNT > 0 else hw_sm_count

    print("╔" + "═" * 68 + "╗")
    print("║  融合 Attention + MoE Megakernel — CUDA 端到端性能测试           ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  GPU:           {props.name}")
    print(f"  HW SM Count:   {hw_sm_count}")
    print(f"  Kernel SMs:    {kernel_sm_count} (compiled)")
    print(f"  VRAM:          {props.total_memory // 1024**3} GB")
    print(f"  CUDA:          {torch.version.cuda}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  Model:         Llama-1B MoE (hidden={HIDDEN_DIM}, inter={INTERMEDIATE_DIM})")
    print(f"  Experts:       {NUM_EXPERTS} total, top-{NUM_EXPERTS_PER_TOK}")
    print(f"  Attention:     {NUM_ATTENTION_HEADS} Q heads, {NUM_KV_HEADS} KV heads, head_dim={HEAD_DIM}")
    print()

    # Load kernel
    mk_func = load_kernel()
    print("  ✓ CUDA kernel 加载成功")
    print()

    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096] if args.sweep else [args.seq_len]

    # Results table
    results = []

    for seq_len in seq_lens:
        print("━" * 70)
        print(f"  Seq Length = {seq_len}")
        print("━" * 70)

        num_kv_blocks = math.ceil(seq_len / KV_BLOCK_SIZE)
        t = create_tensors(seq_len, kernel_sm_count, args.batch_size, device)

        # ---- Correctness ----
        if not args.skip_correctness:
            print("\n  📋 正确性验证...")
            attn_insts = create_attn_only_instructions(kernel_sm_count, layer_idx=0)
            inst_t, tim_t = tensorize(attn_insts, kernel_sm_count, device)
            t['barriers'].zero_()
            t['attn_out'].zero_()
            call_kernel(mk_func, t, inst_t, tim_t, seq_len, args.batch_size)
            torch.cuda.synchronize()

            ref = pytorch_attention(t['q_post_rope'], t['k_cache'], t['v_cache'],
                                    seq_len, 0, num_kv_blocks)
            cuda_out = t['attn_out']
            diff = (cuda_out.float() - ref.float()).abs()
            max_diff = diff.max().item()
            cos_sim = F.cosine_similarity(cuda_out.float().unsqueeze(0),
                                          ref.float().unsqueeze(0)).item()
            status = "✓" if max_diff < 1.0 else "✗"
            print(f"     Attention: max_abs_diff={max_diff:.4e}, cos_sim={cos_sim:.6f} {status}")

            # 检查输出非零
            out_norm = cuda_out.float().norm().item()
            ref_norm = ref.float().norm().item()
            if out_norm < 1e-6:
                print(f"     ⚠️ CUDA 输出接近零 (norm={out_norm:.4e})，可能 kernel 未正确执行")
            print()

        # ---- Benchmark: Attention Only ----
        attn_insts = create_attn_only_instructions(kernel_sm_count, layer_idx=0)
        inst_attn, tim_attn = tensorize(attn_insts, kernel_sm_count, device)

        def run_attn():
            t['barriers'].zero_()
            t['attn_out'].zero_()
            call_kernel(mk_func, t, inst_attn, tim_attn, seq_len, args.batch_size)

        r_attn = benchmark(run_attn, args.warmup, args.iters)
        print(f"  ⏱  Attention Only:       avg={fmt(r_attn['avg']):>12s}  "
              f"p50={fmt(r_attn['p50']):>12s}  std={fmt(r_attn['std']):>10s}")

        # ---- Benchmark: Fused Attention + MoE ----
        fused_insts = create_fused_instructions(kernel_sm_count, layer_idx=0, batch_size=args.batch_size)
        inst_fused, tim_fused = tensorize(fused_insts, kernel_sm_count, device)
        n_fused = sum(1 for inst in fused_insts if inst[5] >= 0)  # moe_token_idx >= 0

        def run_fused():
            t['barriers'].zero_()
            t['attn_out'].zero_()
            t['moe_output_accumulator'].zero_()
            t['moe_intermediate'].zero_()
            t['attn_done_barrier'].zero_()
            call_kernel(mk_func, t, inst_fused, tim_fused, seq_len, args.batch_size)

        r_fused = benchmark(run_fused, args.warmup, args.iters)
        print(f"  ⏱  Fused (Attn+MoE):     avg={fmt(r_fused['avg']):>12s}  "
              f"p50={fmt(r_fused['p50']):>12s}  std={fmt(r_fused['std']):>10s}")

        # ---- Benchmark: PyTorch Baseline ----
        if not args.skip_baseline:
            def run_baseline():
                pytorch_attention(t['q_post_rope'], t['k_cache'], t['v_cache'],
                                  seq_len, 0, num_kv_blocks)
                pytorch_moe(t['moe_input_activations'][0],
                            t['moe_up_weights'], t['moe_gate_weights'], t['moe_down_weights'],
                            t['moe_expert_indices'][0], t['moe_expert_routing_weights'][0], 0)

            r_base = benchmark(run_baseline, args.warmup, args.iters)
            print(f"  ⏱  PyTorch Sequential:   avg={fmt(r_base['avg']):>12s}  "
                  f"p50={fmt(r_base['p50']):>12s}  std={fmt(r_base['std']):>10s}")

        # ---- Summary ----
        moe_overhead_us = r_fused['avg'] - r_attn['avg']
        overlap_eff = (1 - moe_overhead_us / r_fused['avg']) * 100 if r_fused['avg'] > 0 else 0
        tps_fused = args.batch_size / (r_fused['avg'] / 1e6) if r_fused['avg'] > 0 else 0
        tps_attn = args.batch_size / (r_attn['avg'] / 1e6) if r_attn['avg'] > 0 else 0

        print()
        print(f"  📊 分析:")
        print(f"     Attn-only:      {r_attn['avg']:>10.1f} µs  →  {tps_attn:>12,.0f} tok/s")
        print(f"     Fused:          {r_fused['avg']:>10.1f} µs  →  {tps_fused:>12,.0f} tok/s")
        print(f"     MoE 额外开销:   {moe_overhead_us:>10.1f} µs  "
              f"({moe_overhead_us/max(r_fused['avg'],1)*100:.1f}% of fused)")
        print(f"     Overlap 效率:   {overlap_eff:.1f}%")
        print(f"     指令: {len(fused_insts)} total ({n_fused} fused, "
              f"{len(fused_insts)-n_fused} attn-only)")

        if not args.skip_baseline:
            speedup = r_base['avg'] / r_fused['avg'] if r_fused['avg'] > 0 else 0
            tps_base = args.batch_size / (r_base['avg'] / 1e6) if r_base['avg'] > 0 else 0
            print(f"     Baseline:       {r_base['avg']:>10.1f} µs  →  {tps_base:>12,.0f} tok/s")
            print(f"     加速比:         {speedup:.2f}× vs PyTorch baseline")
            results.append((seq_len, r_attn['avg'], r_fused['avg'], r_base['avg'], speedup,
                            tps_fused, moe_overhead_us))
        else:
            results.append((seq_len, r_attn['avg'], r_fused['avg'], 0, 0,
                            tps_fused, moe_overhead_us))

        # 数据量分析
        kv_bytes = seq_len * HEAD_DIM * 2 * 2 * NUM_KV_HEADS
        moe_bytes = (2 * INTERMEDIATE_DIM * HIDDEN_DIM * 2 +
                     HIDDEN_DIM * INTERMEDIATE_DIM * 2) * NUM_EXPERTS_PER_TOK
        print(f"     KV cache:       {kv_bytes/1024:.1f} KB")
        print(f"     MoE weights:    {moe_bytes/1024/1024:.1f} MB")
        print()

    # ---- Summary Table ----
    if len(results) > 1:
        print("\n" + "═" * 90)
        print("  汇总表 (所有 seq_len)")
        print("═" * 90)
        print(f"  {'SeqLen':>7s}  {'Attn(µs)':>10s}  {'Fused(µs)':>10s}  "
              f"{'Base(µs)':>10s}  {'Speedup':>8s}  {'tok/s':>12s}  {'MoE OH(µs)':>11s}")
        print("  " + "-" * 82)
        for sl, attn, fused, base, spd, tps, oh in results:
            base_str = f"{base:10.1f}" if base > 0 else f"{'N/A':>10s}"
            spd_str = f"{spd:7.2f}×" if spd > 0 else f"{'N/A':>8s}"
            print(f"  {sl:>7d}  {attn:>10.1f}  {fused:>10.1f}  "
                  f"{base_str}  {spd_str}  {tps:>12,.0f}  {oh:>11.1f}")
        print()


if __name__ == "__main__":
    main()
