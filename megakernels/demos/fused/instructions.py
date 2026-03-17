"""
Fused Attention + MoE Instruction Definitions

This module defines the instruction types and globals for the fused
Attention + MoE megakernel. The key insight is that each instruction
encodes BOTH an attention task AND an MoE task, allowing the SM to
execute them concurrently via warp specialization.
"""

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor

from megakernels.instructions import BaseGlobals, Instruction


@dataclass
class FusedGlobals(BaseGlobals):
    """
    Extended globals for the fused Attention + MoE kernel.
    Includes all standard model buffers plus fused-specific fields.
    """
    # Standard activation buffers
    post_ln_rope_q: Tensor
    attn_out: Tensor
    attn_lse_intermediates: Tensor
    attn_out_intermediates: Tensor

    # MoE weights
    moe_up_proj_weights: Optional[Tensor] = None
    moe_gate_proj_weights: Optional[Tensor] = None
    moe_down_proj_weights: Optional[Tensor] = None

    # MoE routing
    moe_expert_indices: Optional[Tensor] = None
    moe_expert_weights: Optional[Tensor] = None

    # MoE intermediate buffer
    moe_intermediate: Optional[Tensor] = None

    # MoE configuration
    num_experts: int = 8
    num_experts_per_tok: int = 2

    # Fused-specific fields
    attn_done_barrier: Optional[Tensor] = None  # [max_batch_size] int32
    moe_input_activations: Optional[Tensor] = None  # [max_batch_size, hidden_dim] bf16
    moe_output_accumulator: Optional[Tensor] = None  # [max_batch_size, hidden_dim] fp32

    # Block sizes
    attn_kv_block_size: int = 16
    moe_block_size: int = 16
    matvec_reduction_size: int = 2048
    qkv_block_size: int = 16
    attn_reduction_size: int = 4

    # Batch info
    batch_size: int = 1

    skip_attn_reduction: bool = True


# ============================================================================
# Fused instruction: encodes both attention and MoE tasks
# ============================================================================

@dataclass
class FusedAttnMoE(Instruction):
    """
    Fused Attention + MoE instruction.
    
    This single instruction tells an SM to:
    1. Compute partial Flash Attention for the specified KV head
    2. Simultaneously compute MoE for a previously-completed token
    
    Instruction format (32 ints):
      [0] opcode = 9
      [1] layer_idx
      [2] kv_head_idx (for attention)
      [3] num_partials (attention partitioning)
      [4] partial_idx (which partition this SM handles)
      [5] moe_token_idx (-1 if no MoE work)
      [6] moe_expert_idx (expert index in top-k)
      [7] moe_weight_type (0=up, 1=gate, 2=down)
      [8] moe_start_block
      [9] moe_end_block
      [10] moe_reduction_block
    """
    # Attention fields
    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    # MoE fields (optional, -1 means no MoE work)
    moe_token_idx: int = -1
    moe_expert_idx: int = 0
    moe_weight_type: int = 0  # 0=up, 1=gate, 2=down
    moe_start_block: int = 0
    moe_end_block: int = 0
    moe_reduction_block: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> int:
        return 1  # Depends on QKV computation

    @classmethod
    def tags(cls) -> dict:
        return {"pool": "fused_attn_moe"}

    def has_moe_work(self) -> bool:
        return self.moe_token_idx >= 0

    def cost(self, globs: FusedGlobals):
        # Attention cost
        seq_len = globs.pos_id + 1
        loaded_seq_len = seq_len / self.num_partials
        attn_cost = loaded_seq_len * globs.head_dim * 2

        # MoE cost (if applicable)
        moe_cost = 0
        if self.has_moe_work():
            num_blocks = self.moe_end_block - self.moe_start_block
            if self.moe_weight_type in (0, 1):
                moe_cost = num_blocks * globs.moe_block_size * globs.hidden_size
            else:
                moe_cost = num_blocks * globs.moe_block_size * globs.intermediate_size

        # The effective cost is the MAX of the two (since they overlap)
        return max(attn_cost, moe_cost)


# ============================================================================
# Standalone attention/MoE instructions (for non-fused fallback)
# ============================================================================

@dataclass
class PartialAttentionOnly(Instruction):
    """Standalone partial attention (no MoE overlap)."""
    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return 1

    def cost(self, globs):
        seq_len = globs.pos_id + 1
        loaded_seq_len = seq_len / self.num_partials
        return loaded_seq_len * globs.head_dim * 2


@dataclass
class MoEExpertOnly(Instruction):
    """Standalone MoE expert computation (no attention overlap)."""
    layer_idx: int
    expert_idx: int
    weight_type: int  # 0=up, 1=gate, 2=down
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> int:
        return 3  # After attention reduction

    def cost(self, globs):
        num_blocks = self.end_block_idx - self.start_block_idx
        if self.weight_type in (0, 1):
            return num_blocks * globs.moe_block_size * globs.hidden_size
        else:
            return num_blocks * globs.moe_block_size * globs.intermediate_size
