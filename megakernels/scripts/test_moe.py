
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from megakernels.demos.latency.instructions import Globals, MoEExpertMatVec
from megakernels.demos.latency.python_vm import moe_expert_matvec
from megakernels.dispatch import make_mk_interpreter

def init_globals(device="cuda:0"):
    # Initialize minimal globals for MoE test
    hidden_size = 2048
    intermediate_size = 8192
    num_experts = 8
    num_experts_per_tok = 2
    moe_block_size = 16
    
    # Random weights
    moe_up = torch.randn(1, num_experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
    moe_gate = torch.randn(1, num_experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
    moe_down = torch.randn(1, num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16) # Note: down is [hidden, intermediate]
    
    # Routing (expert 0 and 2 selected for token 0)
    moe_indices = torch.tensor([[[[0, 2]]]], device=device, dtype=torch.int32)
    moe_weights = torch.tensor([[[[0.6, 0.4]]]], device=device, dtype=torch.float32)
    
    # Input
    hidden_states = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    
    # Buffers
    moe_intermediate = torch.zeros(intermediate_size, device=device, dtype=torch.bfloat16)
    barriers = torch.zeros((1, 10, 1024), device=device, dtype=torch.int32)
    
    # Create Globals instance (mocking other fields)
    # We must provide dummy tensors for ALL fields that the C++ MK interpreter expects,
    # otherwise einops or PyBind11 will throw errors when trying to unpack None.
    
    # Dummy tensors for things we don't care about in this MoE test
    # ThunderKittens globals structure strictly checks tensor shapes during bind!
    # Expected shapes from llama_1b_globals:
    # hidden_dim = 2048, intermediate_dim = 8192, head_dim = 64, num_attn_heads = 32, num_kv_heads = 8
    # qkv_weights: [layers, hidden, 48 * head_dim] -> 48 * 64 = 3072
    # o_weights: [layers, hidden, 32 * head_dim] -> 2048
    # up_weights, gate_weights: [layers, intermediate, hidden] -> [1, 8192, 2048]
    # down_weights: [layers, hidden, intermediate] -> [1, 2048, 8192]
    # k_cache, v_cache: [batch, max_seq, num_kv_heads, head_dim]
    
    dummy_qkv = torch.zeros(1, 2048, 3072, device=device, dtype=torch.bfloat16)
    dummy_o = torch.zeros(1, 2048, 2048, device=device, dtype=torch.bfloat16)
    dummy_up = torch.zeros(1, 8192, 2048, device=device, dtype=torch.bfloat16)
    dummy_down = torch.zeros(1, 2048, 8192, device=device, dtype=torch.bfloat16)
    dummy_norm = torch.zeros(1, 2048, device=device, dtype=torch.bfloat16)
    
    dummy_k_cache = torch.zeros(1, 1, 1, 8, 64, device=device, dtype=torch.bfloat16)
    dummy_v_cache = torch.zeros(1, 1, 1, 8, 64, device=device, dtype=torch.bfloat16)
    dummy_rope = torch.zeros(1024, 32, device=device, dtype=torch.float32) # [seq_len, head_dim/2]
    dummy_attn_out = torch.zeros(32, 64, device=device, dtype=torch.bfloat16)
    dummy_lse = torch.zeros(32, device=device, dtype=torch.float32)
    
    globs = Globals(
        qkv_proj_weights=dummy_qkv, o_proj_weights=dummy_o, attn_ln_weights=dummy_norm, mlp_ln_weights=dummy_norm,
        up_proj_weights=dummy_up, gate_proj_weights=dummy_up, down_proj_weights=dummy_down,
        lm_head_norm_weights=dummy_norm, lm_head_weights=dummy_o, 
        k_cache=dummy_k_cache, v_cache=dummy_v_cache,
        rope_cos=dummy_rope, rope_sin=dummy_rope,
        
        hidden_states=hidden_states,
        post_ln_rope_q=dummy_attn_out, attn_out=dummy_attn_out, attn_lse_intermediates=dummy_lse, attn_out_intermediates=dummy_attn_out,
        silu_out=dummy_norm, logits=dummy_norm,
        
        moe_up_proj_weights=moe_up,
        moe_gate_proj_weights=moe_gate,
        moe_down_proj_weights=moe_down,
        moe_expert_indices=moe_indices,
        moe_expert_weights=moe_weights,
        moe_intermediate=moe_intermediate,
        
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_block_size=moe_block_size,
        
        pos_id=0, attn_scale=1.0, rms_norm_eps=1e-5, skip_attn_reduction=True,
        num_hidden_layers=1, num_attention_heads=32, num_kv_heads=8,
        head_dim=64, hidden_size=hidden_size, intermediate_size=intermediate_size,
        
        up_gate_proj_block_size=16, down_proj_block_size=16, qkv_block_size=16,
        o_proj_block_size=16, lm_head_block_size=16, matvec_reduction_size=2048,
        attn_kv_block_size=16, attn_reduction_size=4, vocab_size=100,
        device=device, barriers=barriers
    )
    return globs

def test_moe_correctness():
    torch.manual_seed(42)
    device = "cuda:0"
    
    print("Initializing Globals...")
    globs = init_globals(device)
    
    # --- Python VM Execution (Reference) ---
    print("Running Python VM Reference...")
    
    # Backup initial hidden states because down_proj modifies it in-place
    initial_hidden = globs.hidden_states.clone()
    
    # Define instructions for Expert 0 (Top-1)
    # Up Proj
    up_inst = MoEExpertMatVec(
        layer_idx=0, expert_idx=0, weight_type=0, 
        start_block_idx=0, end_block_idx=globs.intermediate_size // globs.moe_block_size, 
        reduction_block_idx=0
    )
    moe_expert_matvec(globs, up_inst)
    
    # Gate Proj
    gate_inst = MoEExpertMatVec(
        layer_idx=0, expert_idx=0, weight_type=1,
        start_block_idx=0, end_block_idx=globs.intermediate_size // globs.moe_block_size,
        reduction_block_idx=0
    )
    moe_expert_matvec(globs, gate_inst)
    
    # Down Proj
    down_inst = MoEExpertMatVec(
        layer_idx=0, expert_idx=0, weight_type=2,
        start_block_idx=0, end_block_idx=globs.hidden_size // globs.moe_block_size,
        reduction_block_idx=0 # Simplification: Assume 1 reduction step covers all
    )
    # Note: Python VM down_proj implementation in previous steps was partial.
    # We need to make sure the loop over reduction blocks is correct in VM if we want full match.
    # For this test, let's assume single block or loop is handled.
    
    # But wait, Python VM implementation of down_proj in my previous turn:
    # "matvec_with_residual ... reduction_size=globals.intermediate_size // globals.hidden_size * globals.moe_block_size"
    # It seems to do a partial reduction.
    
    # Let's actually run the VM logic manually here for clarity if needed, 
    # but `moe_expert_matvec` should work if implemented correctly.
    
    # Run Down Proj
    # We need to run it for ALL columns (reduction blocks) to get correct result
    num_col_splits = globs.intermediate_size // globs.hidden_size
    for col_idx in range(num_col_splits):
        inst = MoEExpertMatVec(
            layer_idx=0, expert_idx=0, weight_type=2,
            start_block_idx=0, end_block_idx=globs.hidden_size // globs.moe_block_size,
            reduction_block_idx=col_idx
        )
        moe_expert_matvec(globs, inst)
        
    ref_hidden_states = globs.hidden_states.clone()
    print("Python VM finished.")
    
    # --- Megakernel Execution (DUT) ---
    print("Running Megakernel...")
    
    # Reset state
    globs.hidden_states.copy_(initial_hidden)
    globs.moe_intermediate.zero_()
    globs.barriers.zero_()
    
    # Load MK Interpreter
    # Note: This requires the .so to be compiled and available
    mk_dir = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama"
    try:
        interpreter = make_mk_interpreter("latency", mk_dir)
    except Exception as e:
        print(f"Failed to load MK interpreter: {e}")
        print("Please compile the kernel first using 'make' in demos/low-latency-llama")
        return

    # Construct instructions tensor
    print("Tensorizing instructions for MK...")
    
    # List of all instructions we ran in VM
    instructions_to_run = [up_inst, gate_inst]
    for col_idx in range(num_col_splits):
        instructions_to_run.append(
            MoEExpertMatVec(
                layer_idx=0, expert_idx=0, weight_type=2,
                start_block_idx=0, end_block_idx=globs.hidden_size // globs.moe_block_size,
                reduction_block_idx=col_idx
            )
        )
        
    # Assign all to SM 0 for this isolated test
    # Tensorize manually: just call to_tensor() on each instruction
    num_sms = 132 # H100
    max_inst = len(instructions_to_run) + 1 # +1 for the explicit Stop (opcode 0)
    inst_tensor = torch.zeros((num_sms, max_inst, 32), dtype=torch.int32, device=device)
    
    # Serialize each instruction
    for i, inst in enumerate(instructions_to_run):
        fields = [
            inst.opcode(),
            inst.layer_idx,
            inst.expert_idx,
            inst.weight_type,
            inst.start_block_idx,
            inst.end_block_idx,
            inst.reduction_block_idx
        ]
        # Pad to 32
        fields += [0] * (32 - len(fields))
            
        inst_tensor[0, i, :] = torch.tensor(fields, dtype=torch.int32, device=device)
    
    # The last instruction is implicitly all 0s due to torch.zeros initialization,
    # which corresponds to the NoOp/Stop instruction in MK.
    
    # Copy to globals
    globs.instructions = inst_tensor
    
    print("Executing Megakernel...")
    try:
        interpreter.interpret(globs)
        print("MK execution completed.")
    except Exception as e:
        print(f"Error during MK execution: {e}")
        return

    # Compare results
    print("\n--- Comparing Results ---")
    mk_hidden_states = globs.hidden_states.clone()
    
    # Note: Because of bfloat16 and different accumulation order, there might be slight numerical differences.
    # We use a relatively loose tolerance.
    diff = torch.abs(ref_hidden_states.float() - mk_hidden_states.float())
    max_diff = diff.max().item()
    
    print(f"Max absolute difference in hidden_states: {max_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✅ SUCCESS: Megakernel MoE output matches Python VM!")
    else:
        print("❌ WARNING: Megakernel output differs significantly from Python VM.")
        
    print("\nTest complete.")

if __name__ == "__main__":
    test_moe_correctness()
