"""
MK Interpreter for the fused Attention + MoE kernel.

Bridges the Python scheduler output to the CUDA megakernel invocation.
"""

import torch
from einops import rearrange

from megakernels.demos.fused.instructions import FusedGlobals
from megakernels.mk import MK_Interpreter


def interpret_with_mk(
    globs: FusedGlobals,
    mk_func,
):
    """Launch the fused megakernel with the given globals."""
    fourD_k_cache = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v_cache = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    mk_func(
        # VM infrastructure
        globs.barriers,
        globs.instructions,
        globs.timings,
        # Model weights
        globs.qkv_proj_weights,
        globs.attn_ln_weights,
        globs.o_proj_weights,
        globs.mlp_ln_weights,
        # KV cache
        fourD_k_cache,
        fourD_v_cache,
        # Rope
        globs.rope_cos,
        globs.rope_sin,
        # Activation buffers
        globs.hidden_states,
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        # MoE weights
        globs.moe_up_proj_weights,
        globs.moe_gate_proj_weights,
        globs.moe_down_proj_weights,
        # MoE routing
        globs.moe_expert_indices,
        globs.moe_expert_weights,
        # MoE intermediate
        globs.moe_intermediate,
        # Fused-specific
        globs.attn_done_barrier,
        globs.moe_input_activations,
        globs.moe_output_accumulator,
        # Scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.batch_size,
        0,  # current_token_idx (set per-instruction in the kernel)
        -1,  # moe_token_idx (set per-instruction in the kernel)
        stream=torch.cuda.current_stream(),
    )


class FusedMK_Interpreter(MK_Interpreter):
    """Interpreter that dispatches to the compiled fused megakernel."""

    def interpret(self, globs: FusedGlobals):
        interpret_with_mk(globs, self.mk_func)
