"""
Fused Attention + MoE Scheduler

This scheduler generates instructions that OVERLAP attention and MoE computation
on the same SM via warp specialization. The key scheduling strategy:

1. For layer L, we schedule attention for all KV heads (as partial attention ops)
2. For tokens that have already completed attention (from a previous layer or 
   previous batch iteration), we co-schedule MoE work alongside the attention
3. The fused instruction encodes BOTH the attention partition AND the MoE work

Scheduling Algorithm:
  - Attention partials are distributed across SMs (as in the existing scheduler)
  - MoE expert work is split into chunks and paired with attention partials
  - Each SM gets a FusedAttnMoE instruction containing both tasks
  - The MoE work targets tokens from layer L-1 (pipeline across layers)

Example timeline for batch_size=4, 2 layers:
  Layer 0: Token 0,1,2,3 attention (no MoE overlap - first layer)
  Layer 1: Token 0,1,2,3 attention + Token 0,1,2,3 MoE from Layer 0
"""

import math
from typing import Optional

import torch

from megakernels.demos.fused.instructions import (
    FusedAttnMoE,
    FusedGlobals,
    MoEExpertOnly,
    PartialAttentionOnly,
)
from megakernels.instructions import Instruction, NoOp
from megakernels.llama import LlamaForCausalLM
from megakernels.scheduler import DAG_Node, Schedule, ScheduleBuilder
from megakernels.utils import assert_div, get_sm_count


def make_fused_globals(
    model: LlamaForCausalLM,
    batch_size: int = 1,
    skip_attn_reduction: bool = True,
) -> FusedGlobals:
    """Create globals for the fused Attention + MoE kernel."""
    config = model.config
    device = model.device
    dtype = model.dtype

    def make_buffer(shape, buffer_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params
    sm_count = get_sm_count(device)
    max_attn_partitions = sm_count

    barriers = torch.zeros(
        [
            config.num_hidden_layers,
            12,  # increased for fused opcode
            config.num_attention_heads + config.num_key_value_heads * 2,
        ],
        dtype=torch.int32,
        device=device,
    )

    return FusedGlobals(
        # Model params
        qkv_proj_weights=stacked_params.qkv_proj,
        o_proj_weights=stacked_params.o_proj,
        attn_ln_weights=stacked_params.attn_ln_weight,
        mlp_ln_weights=stacked_params.mlp_ln_weight,
        up_proj_weights=stacked_params.up_proj,
        gate_proj_weights=stacked_params.gate_proj,
        down_proj_weights=stacked_params.down_proj,
        lm_head_norm_weights=model.lm_head.input_norm.weight,
        lm_head_weights=model.lm_head.lm_head.weight,
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # Activation buffers
        hidden_states=make_buffer(config.hidden_size),
        post_ln_rope_q=make_buffer(config.hidden_size),
        attn_out=make_buffer(config.hidden_size),
        attn_out_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions, config.head_dim],
            buffer_dtype=torch.float32,
        ),
        attn_lse_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions],
            buffer_dtype=torch.float32,
        ),
        # MoE weights (if model has MoE)
        moe_up_proj_weights=getattr(stacked_params, 'moe_up_proj', None),
        moe_gate_proj_weights=getattr(stacked_params, 'moe_gate_proj', None),
        moe_down_proj_weights=getattr(stacked_params, 'moe_down_proj', None),
        moe_expert_indices=make_buffer(
            [config.num_hidden_layers, 2],  # top-2 experts
            buffer_dtype=torch.int32,
        ),
        moe_expert_weights=make_buffer(
            [config.num_hidden_layers, 2],
            buffer_dtype=torch.float32,
        ),
        moe_intermediate=make_buffer(config.intermediate_size),
        num_experts=getattr(config, 'num_experts', 8),
        num_experts_per_tok=getattr(config, 'num_experts_per_tok', 2),
        # Fused-specific buffers
        attn_done_barrier=torch.zeros(
            batch_size, dtype=torch.int32, device=device
        ),
        moe_input_activations=make_buffer(
            [batch_size, config.hidden_size],
        ),
        moe_output_accumulator=make_buffer(
            [batch_size, config.hidden_size],
            buffer_dtype=torch.float32,
        ),
        batch_size=batch_size,
        # Scalars
        pos_id=0,
        attn_scale=1 / math.sqrt(config.head_dim),
        rms_norm_eps=config.rms_norm_eps,
        skip_attn_reduction=skip_attn_reduction,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        device=device,
        barriers=barriers,
    )


def schedule_fused_attn_moe_layer(
    globs: FusedGlobals,
    layer_idx: int,
    moe_ready_tokens: list[int],
    num_attention_partitions: int = 1,
) -> list[FusedAttnMoE]:
    """
    Schedule a single layer with fused Attention + MoE instructions.
    
    Args:
        globs: Global configuration
        layer_idx: Which layer we're processing
        moe_ready_tokens: List of token indices whose attention is complete
                          and are ready for MoE computation
        num_attention_partitions: How many SMs to split each KV head's
                                  attention across
    
    Returns:
        List of FusedAttnMoE instructions, one per SM
    """
    sm_count = globs.sm_count()
    instructions: list[FusedAttnMoE] = []
    
    # ---- Generate attention tasks ----
    # Each KV head needs `num_attention_partitions` SMs
    attn_tasks = []
    for kv_head_idx in range(globs.num_kv_heads):
        for partial_idx in range(num_attention_partitions):
            attn_tasks.append({
                'layer_idx': layer_idx,
                'kv_head_idx': kv_head_idx,
                'num_partials': num_attention_partitions,
                'partial_idx': partial_idx,
            })
    
    # ---- Generate MoE tasks ----
    moe_tasks = []
    if moe_ready_tokens and globs.moe_up_proj_weights is not None:
        for token_idx in moe_ready_tokens:
            for expert_idx in range(globs.num_experts_per_tok):
                # Up projection
                num_up_blocks = globs.intermediate_size // globs.moe_block_size
                blocks_per_chunk = max(1, num_up_blocks // (sm_count // 4))
                for start in range(0, num_up_blocks, blocks_per_chunk):
                    end = min(start + blocks_per_chunk, num_up_blocks)
                    moe_tasks.append({
                        'moe_token_idx': token_idx,
                        'moe_expert_idx': expert_idx,
                        'moe_weight_type': 0,  # up
                        'moe_start_block': start,
                        'moe_end_block': end,
                        'moe_reduction_block': 0,
                    })
                    # Gate projection (same blocks)
                    moe_tasks.append({
                        'moe_token_idx': token_idx,
                        'moe_expert_idx': expert_idx,
                        'moe_weight_type': 1,  # gate
                        'moe_start_block': start,
                        'moe_end_block': end,
                        'moe_reduction_block': 0,
                    })
                
                # Down projection
                num_down_blocks = globs.hidden_size // globs.moe_block_size
                num_col_splits = globs.intermediate_size // globs.hidden_size
                for col_idx in range(num_col_splits):
                    blocks_per_chunk = max(1, num_down_blocks // (sm_count // 4))
                    for start in range(0, num_down_blocks, blocks_per_chunk):
                        end = min(start + blocks_per_chunk, num_down_blocks)
                        moe_tasks.append({
                            'moe_token_idx': token_idx,
                            'moe_expert_idx': expert_idx,
                            'moe_weight_type': 2,  # down
                            'moe_start_block': start,
                            'moe_end_block': end,
                            'moe_reduction_block': col_idx,
                        })
    
    # ---- Pair attention tasks with MoE tasks ----
    # Strategy: round-robin assign MoE tasks to attention tasks
    num_attn = len(attn_tasks)
    num_moe = len(moe_tasks)
    
    for i, attn_task in enumerate(attn_tasks):
        # Pair with a MoE task if available
        if i < num_moe:
            moe_task = moe_tasks[i]
        else:
            moe_task = {
                'moe_token_idx': -1,
                'moe_expert_idx': 0,
                'moe_weight_type': 0,
                'moe_start_block': 0,
                'moe_end_block': 0,
                'moe_reduction_block': 0,
            }
        
        instructions.append(FusedAttnMoE(
            **attn_task,
            **moe_task,
        ))
    
    # If there are more MoE tasks than attention tasks,
    # create standalone MoE-only instructions (with dummy attention)
    for i in range(num_attn, num_moe):
        moe_task = moe_tasks[i]
        # Use a dummy attention task (partition 0 of head 0, which is already done)
        instructions.append(FusedAttnMoE(
            layer_idx=layer_idx,
            kv_head_idx=0,
            num_partials=num_attention_partitions,
            partial_idx=0,  # Will be de-duplicated in the kernel
            **moe_task,
        ))
    
    return instructions


def make_fused_dag(
    globs: FusedGlobals,
    layer_limit: Optional[int] = None,
) -> tuple[list[DAG_Node], DAG_Node]:
    """
    Build a DAG of fused Attention + MoE instructions.
    
    The key optimization: for layer L > 0, we overlap attention for layer L
    with MoE computation for tokens processed in layer L-1.
    """
    nodes: list[DAG_Node] = []
    
    if layer_limit is not None:
        nlayers = layer_limit
    else:
        nlayers = globs.num_hidden_layers
    
    last_outputs: list[DAG_Node] = []
    moe_ready_tokens: list[int] = []
    
    for layer_idx in range(nlayers):
        # For layer 0: no MoE overlap (no tokens have completed attention yet)
        # For layer > 0: overlap MoE for all tokens from the previous layer
        
        fused_instructions = schedule_fused_attn_moe_layer(
            globs=globs,
            layer_idx=layer_idx,
            moe_ready_tokens=moe_ready_tokens,
            num_attention_partitions=1,  # Single partition for decode
        )
        
        layer_nodes = []
        for inst in fused_instructions:
            node = DAG_Node(inst, last_outputs)
            layer_nodes.append(node)
        
        nodes.extend(layer_nodes)
        last_outputs = layer_nodes
        
        # After this layer, all tokens in the batch are ready for MoE
        moe_ready_tokens = list(range(globs.batch_size))
    
    end_node = DAG_Node(NoOp(), last_outputs)
    return nodes, end_node


class FusedScheduleBuilder(ScheduleBuilder):
    """Schedule builder for the fused Attention + MoE kernel."""
    
    @classmethod
    def make_globals(cls, model, batch_size=1):
        return make_fused_globals(model, batch_size=batch_size)
    
    @classmethod
    def make_dag(cls, globs, stop_after_op=None, layer_limit=None):
        return make_fused_dag(globs, layer_limit=layer_limit)
    
    @classmethod
    def build(cls, model, batch_size=1, layer_limit=None):
        globs = cls.make_globals(model, batch_size=batch_size)
        dag_nodes, end_node = cls.make_dag(globs, layer_limit=layer_limit)
        return Schedule(globs, dag_nodes, end_node)
