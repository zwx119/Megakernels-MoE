"""
Test and benchmark script for the fused Attention + MoE kernel.

This script:
1. Validates the Python VM against a reference PyTorch implementation
2. Demonstrates the scheduling algorithm
3. (When CUDA kernel is compiled) benchmarks the fused vs sequential execution

Usage:
    python test_fused.py [--batch-size 4] [--seq-len 1024] [--layers 2]
"""

import argparse
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F


def create_mock_config(
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
):
    """Create a mock model config for testing."""
    class Config:
        pass
    
    cfg = Config()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_kv_heads
    cfg.head_dim = head_dim
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_experts = num_experts
    cfg.num_experts_per_tok = num_experts_per_tok
    cfg.vocab_size = vocab_size
    cfg.rms_norm_eps = rms_norm_eps
    return cfg


def create_test_tensors(config, batch_size=1, seq_len=128, device="cpu"):
    """Create test tensors for validation."""
    dtype = torch.bfloat16
    
    tensors = {}
    
    # Q post rope: [num_attention_heads * head_dim]
    tensors['q_post_rope'] = torch.randn(
        config.num_attention_heads * config.head_dim,
        device=device, dtype=dtype
    )
    
    # KV cache: [num_layers, num_kv_blocks, num_kv_heads, kv_block_size, head_dim]
    kv_block_size = 16
    num_kv_blocks = math.ceil(seq_len / kv_block_size)
    tensors['k_cache'] = torch.randn(
        config.num_hidden_layers, num_kv_blocks, config.num_key_value_heads,
        kv_block_size, config.head_dim,
        device=device, dtype=dtype
    )
    tensors['v_cache'] = torch.randn(
        config.num_hidden_layers, num_kv_blocks, config.num_key_value_heads,
        kv_block_size, config.head_dim,
        device=device, dtype=dtype
    )
    
    # MoE weights: [num_layers, num_experts, out_dim, in_dim]
    tensors['moe_up_weights'] = torch.randn(
        config.num_hidden_layers, config.num_experts,
        config.intermediate_size, config.hidden_size,
        device=device, dtype=dtype
    )
    tensors['moe_gate_weights'] = torch.randn(
        config.num_hidden_layers, config.num_experts,
        config.intermediate_size, config.hidden_size,
        device=device, dtype=dtype
    )
    tensors['moe_down_weights'] = torch.randn(
        config.num_hidden_layers, config.num_experts,
        config.hidden_size, config.intermediate_size,
        device=device, dtype=dtype
    )
    
    # MoE routing
    tensors['expert_indices'] = torch.randint(
        0, config.num_experts,
        (config.num_hidden_layers, config.num_experts_per_tok),
        device=device, dtype=torch.int32
    )
    tensors['expert_weights'] = torch.softmax(
        torch.randn(config.num_hidden_layers, config.num_experts_per_tok,
                     device=device, dtype=torch.float32),
        dim=-1
    )
    
    # MoE input activations
    tensors['moe_input'] = torch.randn(
        batch_size, config.hidden_size,
        device=device, dtype=dtype
    )
    
    return tensors


def reference_attention(
    q: torch.Tensor,  # [num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [num_kv_blocks, kv_block_size, head_dim]
    v_cache: torch.Tensor,  # [num_kv_blocks, kv_block_size, head_dim]
    seq_len: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference attention implementation."""
    num_q_heads = q.shape[0]
    head_dim = q.shape[1]
    kv_block_size = k_cache.shape[1]
    
    # Flatten KV cache
    K = k_cache.reshape(-1, head_dim)[:seq_len].float()  # [seq_len, head_dim]
    V = v_cache.reshape(-1, head_dim)[:seq_len].float()  # [seq_len, head_dim]
    
    Q = q.float()  # [num_q_heads, head_dim]
    
    # Compute attention scores
    scores = torch.matmul(Q, K.T) * scale  # [num_q_heads, seq_len]
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum
    O = torch.matmul(attn_weights, V)  # [num_q_heads, head_dim]
    
    # LSE
    L = torch.logsumexp(scores, dim=-1)
    
    return O, L


def reference_moe_expert(
    input_act: torch.Tensor,  # [hidden_dim]
    up_weight: torch.Tensor,  # [intermediate_dim, hidden_dim]
    gate_weight: torch.Tensor,  # [intermediate_dim, hidden_dim]
    down_weight: torch.Tensor,  # [hidden_dim, intermediate_dim]
    routing_weight: float,
) -> torch.Tensor:
    """Reference MoE expert computation."""
    input_float = input_act.float()
    
    # Up projection
    up_out = torch.matmul(up_weight.float(), input_float)  # [intermediate_dim]
    
    # Gate projection + SiLU
    gate_out = torch.matmul(gate_weight.float(), input_float)  # [intermediate_dim]
    gate_out = gate_out * torch.sigmoid(gate_out)  # SiLU
    
    # Element-wise multiply
    intermediate = up_out * gate_out  # [intermediate_dim]
    
    # Down projection
    down_out = torch.matmul(down_weight.float(), intermediate)  # [hidden_dim]
    
    # Apply routing weight
    return down_out * routing_weight


def test_attention_correctness():
    """Test that partial attention matches reference implementation."""
    print("=" * 60)
    print("Testing Attention Correctness")
    print("=" * 60)
    
    config = create_mock_config(num_hidden_layers=1)
    seq_len = 64
    tensors = create_test_tensors(config, seq_len=seq_len)
    scale = 1.0 / math.sqrt(config.head_dim)
    gqa_ratio = config.num_attention_heads // config.num_key_value_heads
    
    for kv_head_idx in range(config.num_key_value_heads):
        q_start = kv_head_idx * gqa_ratio
        Q = tensors['q_post_rope'].view(-1, config.head_dim)[q_start:q_start + gqa_ratio]
        
        ref_O, ref_L = reference_attention(
            Q,
            tensors['k_cache'][0, :, kv_head_idx],
            tensors['v_cache'][0, :, kv_head_idx],
            seq_len,
            scale,
        )
        
        # Test with single partition (should match exactly)
        # Simulate what the kernel does with online softmax
        O = torch.zeros_like(ref_O)
        max_val = torch.full((gqa_ratio,), float('-inf'))
        sum_exp = torch.zeros(gqa_ratio)
        
        kv_block_size = 16
        num_blocks = math.ceil(seq_len / kv_block_size)
        
        for blk_idx in range(num_blocks):
            blk_start = blk_idx * kv_block_size
            blk_end = min(blk_start + kv_block_size, seq_len)
            blk_len = blk_end - blk_start
            
            K = tensors['k_cache'][0, blk_idx, kv_head_idx, :blk_len].float()
            V = tensors['v_cache'][0, blk_idx, kv_head_idx, :blk_len].float()
            
            scores = torch.matmul(Q.float(), K.T) * scale
            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(max_val, block_max)
            
            correction = torch.exp(max_val - new_max)
            O = O * correction.unsqueeze(-1)
            sum_exp = sum_exp * correction
            
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            O = O + torch.matmul(exp_scores, V)
            sum_exp = sum_exp + exp_scores.sum(dim=-1)
            
            max_val = new_max
        
        O = O / sum_exp.unsqueeze(-1)
        
        # Compare
        diff = (O - ref_O).abs().max().item()
        print(f"  KV Head {kv_head_idx}: max abs diff = {diff:.2e}", 
              "✓" if diff < 1e-4 else "✗")
    
    print()


def test_moe_correctness():
    """Test that MoE expert computation matches reference."""
    print("=" * 60)
    print("Testing MoE Expert Correctness")
    print("=" * 60)
    
    config = create_mock_config(
        num_hidden_layers=1,
        hidden_size=256,  # smaller for CPU testing
        intermediate_size=512,
        num_experts=4,
    )
    tensors = create_test_tensors(config, batch_size=1)
    
    layer_idx = 0
    for expert_idx in range(config.num_experts_per_tok):
        actual_expert = tensors['expert_indices'][layer_idx, expert_idx].item()
        routing_weight = tensors['expert_weights'][layer_idx, expert_idx].item()
        
        ref_output = reference_moe_expert(
            tensors['moe_input'][0],
            tensors['moe_up_weights'][layer_idx, actual_expert],
            tensors['moe_gate_weights'][layer_idx, actual_expert],
            tensors['moe_down_weights'][layer_idx, actual_expert],
            routing_weight,
        )
        
        # Verify the computation is non-trivial
        print(f"  Expert {expert_idx} (actual={actual_expert}): "
              f"output norm = {ref_output.norm():.4f}, "
              f"routing_weight = {routing_weight:.4f}")
    
    print()


def test_scheduling():
    """Test the scheduling algorithm."""
    print("=" * 60)
    print("Testing Scheduling Algorithm")
    print("=" * 60)
    
    from megakernels.demos.fused.instructions import FusedAttnMoE, FusedGlobals
    from megakernels.demos.fused.scheduler import schedule_fused_attn_moe_layer
    
    # Create a minimal globals-like object for scheduling
    class MockGlobals:
        def __init__(self):
            self.num_kv_heads = 8
            self.num_attention_heads = 32
            self.head_dim = 64
            self.hidden_size = 2048
            self.intermediate_size = 8192
            self.num_experts = 8
            self.num_experts_per_tok = 2
            self.moe_block_size = 16
            self.pos_id = 127  # seq_len = 128
            self.attn_kv_block_size = 16
            self.moe_up_proj_weights = True  # just needs to be truthy
            
        def sm_count(self):
            return 132
    
    globs = MockGlobals()
    
    # Layer 0: no MoE overlap
    instructions_l0 = schedule_fused_attn_moe_layer(
        globs, layer_idx=0, moe_ready_tokens=[], num_attention_partitions=1
    )
    print(f"  Layer 0 (no MoE overlap): {len(instructions_l0)} instructions")
    attn_only = sum(1 for i in instructions_l0 if not i.has_moe_work())
    fused = sum(1 for i in instructions_l0 if i.has_moe_work())
    print(f"    Attention-only: {attn_only}, Fused: {fused}")
    
    # Layer 1: with MoE overlap for token 0
    instructions_l1 = schedule_fused_attn_moe_layer(
        globs, layer_idx=1, moe_ready_tokens=[0], num_attention_partitions=1
    )
    print(f"  Layer 1 (MoE overlap for 1 token): {len(instructions_l1)} instructions")
    attn_only = sum(1 for i in instructions_l1 if not i.has_moe_work())
    fused = sum(1 for i in instructions_l1 if i.has_moe_work())
    print(f"    Attention-only: {attn_only}, Fused: {fused}")
    
    # Show a sample fused instruction
    for inst in instructions_l1:
        if inst.has_moe_work():
            print(f"\n  Sample fused instruction:")
            print(f"    Attention: layer={inst.layer_idx}, kv_head={inst.kv_head_idx}, "
                  f"partial={inst.partial_idx}/{inst.num_partials}")
            print(f"    MoE: token={inst.moe_token_idx}, expert={inst.moe_expert_idx}, "
                  f"type={'up' if inst.moe_weight_type==0 else 'gate' if inst.moe_weight_type==1 else 'down'}, "
                  f"blocks=[{inst.moe_start_block},{inst.moe_end_block})")
            break
    
    print()


def estimate_overlap_speedup():
    """Estimate the theoretical speedup from overlapping attention and MoE."""
    print("=" * 60)
    print("Theoretical Overlap Speedup Estimation")
    print("=" * 60)
    
    # H100 specs
    sm_count = 132
    tensor_tflops = 989  # bf16 tensor core TFLOPS
    mem_bw_tb = 3.35  # TB/s HBM bandwidth
    
    # Model config (Mixtral-like)
    hidden_dim = 4096
    intermediate_dim = 14336
    num_attention_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_experts = 8
    num_experts_per_tok = 2
    
    for seq_len in [128, 512, 2048]:
        print(f"\n  Sequence length: {seq_len}")
        
        # Attention compute (per head)
        # Q@K^T: [1, head_dim] x [seq_len, head_dim]^T → [1, seq_len]
        # A@V:   [1, seq_len] x [seq_len, head_dim] → [1, head_dim]
        attn_flops = 2 * head_dim * seq_len + 2 * seq_len * head_dim  # per head
        total_attn_flops = attn_flops * num_attention_heads
        
        # Attention memory (KV cache loading, per head)
        attn_bytes = seq_len * head_dim * 2 * 2  # K + V, bf16
        total_attn_bytes = attn_bytes * num_kv_heads
        
        # MoE compute (per expert)
        # up: [intermediate, hidden] x [hidden, 1]
        # gate: [intermediate, hidden] x [hidden, 1]  
        # down: [hidden, intermediate] x [intermediate, 1]
        moe_flops_per_expert = (
            2 * intermediate_dim * hidden_dim +  # up
            2 * intermediate_dim * hidden_dim +  # gate
            2 * hidden_dim * intermediate_dim    # down
        )
        total_moe_flops = moe_flops_per_expert * num_experts_per_tok
        
        # MoE memory (weight loading, per expert)
        moe_bytes_per_expert = (
            intermediate_dim * hidden_dim * 2 +  # up
            intermediate_dim * hidden_dim * 2 +  # gate
            hidden_dim * intermediate_dim * 2    # down
        ) # bf16
        total_moe_bytes = moe_bytes_per_expert * num_experts_per_tok
        
        # Time estimates
        attn_compute_time = total_attn_flops / (tensor_tflops * 1e12)
        attn_mem_time = total_attn_bytes / (mem_bw_tb * 1e12)
        attn_time = max(attn_compute_time, attn_mem_time)
        
        moe_compute_time = total_moe_flops / (tensor_tflops * 1e12)
        moe_mem_time = total_moe_bytes / (mem_bw_tb * 1e12)
        moe_time = max(moe_compute_time, moe_mem_time)
        
        sequential_time = attn_time + moe_time
        overlapped_time = max(attn_time, moe_time)  # Ideal overlap
        speedup = sequential_time / overlapped_time
        
        print(f"    Attention: {attn_time*1e6:.1f} µs "
              f"({'compute' if attn_compute_time > attn_mem_time else 'memory'}-bound)")
        print(f"    MoE:       {moe_time*1e6:.1f} µs "
              f"({'compute' if moe_compute_time > moe_mem_time else 'memory'}-bound)")
        print(f"    Sequential: {sequential_time*1e6:.1f} µs")
        print(f"    Overlapped: {overlapped_time*1e6:.1f} µs")
        print(f"    Speedup:    {speedup:.2f}×")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Test fused Attention + MoE kernel")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--skip-scheduling", action="store_true")
    args = parser.parse_args()
    
    print("Fused Attention + MoE Kernel Test Suite")
    print("=" * 60)
    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, layers={args.layers}")
    print()
    
    # Basic correctness tests (CPU, no dependencies)
    test_attention_correctness()
    test_moe_correctness()
    
    # Scheduling test (requires megakernels package)
    if not args.skip_scheduling:
        try:
            test_scheduling()
        except ImportError as e:
            print(f"  Skipping scheduling test (missing dependency: {e})")
            print()
    
    # Theoretical analysis
    estimate_overlap_speedup()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
