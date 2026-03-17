"""
Python VM for simulating the fused Attention + MoE kernel.

This allows testing the scheduling logic and verifying correctness
without needing to compile and run the CUDA kernel.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from megakernels.demos.fused.instructions import FusedAttnMoE, FusedGlobals


class FusedAttnMoE_PythonVM:
    """
    Python reference implementation of the fused Attention + MoE kernel.
    Simulates the warp-specialized execution for correctness verification.
    """

    def __init__(self, globs: FusedGlobals):
        self.globs = globs

    def execute_instruction(self, inst: FusedAttnMoE):
        """Execute a single fused instruction (simulating one SM)."""
        g = self.globs

        # ---- Attention computation ----
        attn_output = self._compute_attention(inst)

        # ---- MoE computation (if applicable) ----
        moe_output = None
        if inst.has_moe_work():
            moe_output = self._compute_moe(inst)

        return attn_output, moe_output

    def _compute_attention(self, inst: FusedAttnMoE):
        """
        Compute partial Flash Attention for the specified KV head partition.
        This is the Python reference for what the attention warps do.
        """
        g = self.globs
        layer_idx = inst.layer_idx
        kv_head_idx = inst.kv_head_idx
        gqa_ratio = g.num_attention_heads // g.num_kv_heads

        seq_len = g.pos_id + 1
        total_blocks = math.ceil(seq_len / g.attn_kv_block_size)
        blocks_per_partial = math.ceil(total_blocks / inst.num_partials)
        start_blk = inst.partial_idx * blocks_per_partial
        end_blk = min(start_blk + blocks_per_partial, total_blocks)

        q_head_start = kv_head_idx * gqa_ratio
        head_dim = g.head_dim

        # Get Q vectors for all GQA heads in this group
        Q = g.post_ln_rope_q.view(-1, head_dim)[q_head_start:q_head_start + gqa_ratio]
        # Q shape: [gqa_ratio, head_dim]

        if start_blk >= end_blk:
            return {
                'O': torch.zeros(gqa_ratio, head_dim, device=Q.device, dtype=torch.float32),
                'L': torch.full((gqa_ratio,), float('-inf'), device=Q.device, dtype=torch.float32),
            }

        # Online softmax attention
        O = torch.zeros(gqa_ratio, head_dim, device=Q.device, dtype=torch.float32)
        max_val = torch.full((gqa_ratio,), float('-inf'), device=Q.device, dtype=torch.float32)
        sum_exp = torch.zeros(gqa_ratio, device=Q.device, dtype=torch.float32)

        scale = g.attn_scale

        for blk_idx in range(start_blk, end_blk):
            blk_start = blk_idx * g.attn_kv_block_size
            blk_end = min(blk_start + g.attn_kv_block_size, seq_len)
            blk_len = blk_end - blk_start

            K = g.k_cache[layer_idx, blk_idx, kv_head_idx, :blk_len, :].float()
            V = g.v_cache[layer_idx, blk_idx, kv_head_idx, :blk_len, :].float()

            # Q @ K^T
            attn_scores = torch.matmul(Q.float(), K.T) * scale  # [gqa_ratio, blk_len]

            # Online softmax update
            block_max = attn_scores.max(dim=-1).values  # [gqa_ratio]
            new_max = torch.maximum(max_val, block_max)

            # Correction for previous blocks
            correction = torch.exp(max_val - new_max)
            O = O * correction.unsqueeze(-1)
            sum_exp = sum_exp * correction

            # Current block contribution
            exp_scores = torch.exp(attn_scores - new_max.unsqueeze(-1))  # [gqa_ratio, blk_len]
            O = O + torch.matmul(exp_scores, V)
            sum_exp = sum_exp + exp_scores.sum(dim=-1)

            max_val = new_max

        # Normalize
        O = O / sum_exp.unsqueeze(-1)
        L = torch.log(sum_exp) + max_val  # log-sum-exp

        return {'O': O, 'L': L}

    def _compute_moe(self, inst: FusedAttnMoE):
        """
        Compute MoE expert matvec for the specified token.
        This is the Python reference for what the MoE warps do.
        """
        g = self.globs
        token_idx = inst.moe_token_idx
        expert_idx = inst.moe_expert_idx
        weight_type = inst.moe_weight_type

        # Get the actual expert ID from routing
        actual_expert_id = g.moe_expert_indices[inst.layer_idx, expert_idx].item()

        start_block = inst.moe_start_block
        end_block = inst.moe_end_block
        block_size = g.moe_block_size

        start_row = start_block * block_size
        end_row = end_block * block_size

        if weight_type == 0:  # up_proj
            # input: [hidden_dim], weight: [intermediate_dim, hidden_dim]
            input_act = g.moe_input_activations[token_idx].float()
            weight = g.moe_up_proj_weights[inst.layer_idx, actual_expert_id, start_row:end_row, :].float()
            output = torch.matmul(weight, input_act)
            return {'type': 'up', 'output': output, 'start_row': start_row}

        elif weight_type == 1:  # gate_proj
            input_act = g.moe_input_activations[token_idx].float()
            weight = g.moe_gate_proj_weights[inst.layer_idx, actual_expert_id, start_row:end_row, :].float()
            gate_out = torch.matmul(weight, input_act)
            # SiLU activation
            gate_out = gate_out * torch.sigmoid(gate_out)
            return {'type': 'gate', 'output': gate_out, 'start_row': start_row}

        elif weight_type == 2:  # down_proj
            # input: [intermediate_dim], weight: [hidden_dim, intermediate_dim]
            reduction_block = inst.moe_reduction_block
            col_start = reduction_block * g.hidden_size
            col_end = col_start + g.hidden_size

            input_act = g.moe_intermediate[col_start:col_end].float()
            weight = g.moe_down_proj_weights[inst.layer_idx, actual_expert_id, start_row:end_row, col_start:col_end].float()
            output = torch.matmul(weight, input_act)

            # Apply routing weight
            routing_weight = g.moe_expert_weights[inst.layer_idx, expert_idx].item()
            output = output * routing_weight

            return {'type': 'down', 'output': output, 'start_row': start_row}


def run_fused_python_vm(
    globs: FusedGlobals,
    instructions: list[FusedAttnMoE],
):
    """
    Run all fused instructions through the Python VM.
    
    This simulates the concurrent execution on multiple SMs,
    executing attention and MoE in the order specified by the scheduler.
    """
    vm = FusedAttnMoE_PythonVM(globs)

    attn_results = {}
    moe_results = {}

    for inst in instructions:
        attn_out, moe_out = vm.execute_instruction(inst)

        key = (inst.layer_idx, inst.kv_head_idx, inst.partial_idx)
        attn_results[key] = attn_out

        if moe_out is not None:
            moe_key = (inst.moe_token_idx, inst.moe_expert_idx,
                       inst.moe_weight_type, inst.moe_start_block)
            moe_results[moe_key] = moe_out

    return attn_results, moe_results
