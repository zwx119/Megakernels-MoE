/**
 * ============================================================================
 *   融合 Attention + MoE Megakernel — 主入口文件
 * ============================================================================
 *
 * 【文件说明】
 *   本文件负责：
 *   1. 引入所有算子实现（.cu 文件）
 *   2. 实例化具体的模板参数
 *   3. 通过 pybind11 暴露 Python 接口
 *
 * 【与框架的集成方式】
 *   使用 megakernel::mk<config, globals, ops...> 模板函数作为 kernel 入口。
 *   框架会自动生成 main_loop 并根据 opcode 分派到对应的算子。
 *
 *   关键点：使用 default_config 而非自定义 config，
 *   这样完全复用现有框架，warp 分流在 consumer::run 内部完成。
 *
 * 【使用方法】
 *   编译: cd demos/fused-attn-moe && make
 *   Python: import mk_fused_attn_moe
 */

#include "fused_attn_moe.cuh"
#include "fused_attn_moe.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

// ============================================================================
// 模型参数实例化
// ============================================================================

#define FUSED_NUM_LAYERS 16

// 使用 fused_globals_t 模板，参数与 Llama-1B MoE 配置一致
using fused_globals = fused_globals_t<
    FUSED_NUM_LAYERS,
    FUSED_HIDDEN_DIM,          // 2048
    FUSED_INTERMEDIATE_DIM,    // 8192
    FUSED_HEAD_DIM,            // 64
    FUSED_NUM_ATTENTION_HEADS, // 32
    FUSED_NUM_KV_HEADS,        // 8
    FUSED_KV_BLOCK_SIZE,       // 16
    16,                        // matvec_block_size
#if defined(KITTENS_BLACKWELL)
    148,                       // B200 SM count
#elif defined(FUSED_SM_COUNT)
    FUSED_SM_COUNT,            // 用户自定义 SM count（例如 H20=78）
#else
    132,                       // H100 SM count (默认)
#endif
    FUSED_NUM_EXPERTS,         // 8
    FUSED_NUM_EXPERTS_PER_TOK  // 2
>;

// 【关键】使用 default_config 实例化算子
using fused_op_t = fused_attn_moe_op<default_config, fused_globals>;

// ============================================================================
// Pybind11 模块绑定
// ============================================================================
// 框架通过 kittens::py::bind_kernel 自动生成 Python 绑定。
// mk<config, globals, ops...> 会被编译为 CUDA kernel。
// Python 端传入 tensor 指针，框架自动映射到 globals 的各个字段。

PYBIND11_MODULE(mk_fused_attn_moe, m) {
    m.doc() = "融合 Attention + MoE megakernel (warp specialization)";

    kittens::py::bind_kernel<
        mk<default_config, fused_globals, fused_op_t>>(
        m, "mk_fused_attn_moe",
        // VM 基础设施
        &fused_globals::Bar,
        &fused_globals::instructions,
        &fused_globals::timings,
        // 模型权重
        &fused_globals::qkv_weights,
        &fused_globals::attn_norm_weights,
        &fused_globals::o_weights,
        &fused_globals::mlp_norm_weights,
        // KV cache
        &fused_globals::k_cache,
        &fused_globals::v_cache,
        // RoPE
        &fused_globals::rope_cos,
        &fused_globals::rope_sin,
        // Activation 缓冲区
        &fused_globals::hidden_states,
        &fused_globals::q_post_rope,
        &fused_globals::attn_out,
        &fused_globals::attn_lse_intermediates,
        &fused_globals::attn_out_intermediates,
        // MoE 权重
        &fused_globals::moe_up_weights,
        &fused_globals::moe_gate_weights,
        &fused_globals::moe_down_weights,
        // MoE Routing
        &fused_globals::moe_expert_indices,
        &fused_globals::moe_expert_routing_weights,
        // MoE 中间缓冲
        &fused_globals::moe_intermediate,
        // 融合特有字段
        &fused_globals::attn_done_barrier,
        &fused_globals::moe_input_activations,
        &fused_globals::moe_output_accumulator,
        // 标量
        &fused_globals::pos_id,
        &fused_globals::attn_scale,
        &fused_globals::rms_norm_eps,
        &fused_globals::batch_size,
        &fused_globals::current_token_idx,
        &fused_globals::moe_token_idx,
        &fused_globals::skip_attn_reduction
    );
}
