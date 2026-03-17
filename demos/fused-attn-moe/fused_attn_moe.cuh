#pragma once

/**
 * ============================================================================
 *   融合 Attention + MoE Megakernel 通过 Warp Specialization
 * ============================================================================
 *
 * 【核心思想】
 *   在大模型 decode 阶段，低延迟场景（batch size 较小）下，Flash Attention 和
 *   MoE 计算各自都无法将 SM 的 Tensor Core 完全利用。通过 warp specialization，
 *   在同一个 SM 内，让一部分 consumer warp 计算当前 token 的 attention，同时让
 *   另一部分 consumer warp 计算已完成 attention 的历史 token 的 MoE，从而提高
 *   Tensor Core 利用率。
 *
 * 【架构设计】（基于 Megakernel 框架，每个 SM 20 个 warp）
 *   - Warps 0~7   (8个warp):  Attention consumer warps（计算 Flash Attention）
 *   - Warps 8~15  (8个warp):  MoE consumer warps（计算 MoE expert matvec）
 *   - Warp  16:   Loader（TMA异步加载 KV cache 和 MoE 权重）
 *   - Warp  17:   Storer（将结果写回 global memory）
 *   - Warp  18:   Launcher（Blackwell 上的 MMA tensor core 初始化）
 *   - Warp  19:   Controller（指令获取、信号量初始化、页面分配）
 *
 * 【与 MegaBlocks 的区别】
 *   MegaBlocks 是在 batch 维度做 expert 并行（block-sparse）。
 *   我们是在 SM 内部通过 warp specialization 做 op 级并行——
 *   同一个 SM 同时执行 Attention 和 MoE 两种不同的计算。
 *
 * 【同步机制】
 *   1. Attention warps 完成 token T 的 attention 后，原子递增 attn_done_barrier[T]
 *   2. MoE warps 自旋等待 attn_done_barrier[T-1]，确保历史 token 的 attention 已完成
 *   3. 共享内存信号量（semaphore）协调 TMA 数据传输
 *   4. 页面化的共享内存管理权重 tile 和 activation
 *
 * 【数据流】
 *   Token N:   [Attention Warps 计算 attention] --> signal barrier
 *   Token N-1: [MoE Warps 检查 barrier] --> 计算 MoE（与 Token N 的 attn 重叠执行）
 *
 * 【关键设计决策】
 *   - 复用 Megakernel 框架的 default_config（NUM_CONSUMER_WARPS=16）
 *   - 在 consumer::run 内部根据 warpid() 分流：< 8 走 attn，>= 8 走 MoE
 *   - 共享内存页面布局：page0=QOL（attention），page1=KV cache，page2~12=MoE 权重
 *   - 只用 1 个 attention warp 做 decode attention（单 query，不需要更多）
 *   - MoE 使用 matvec_pipeline 的模式（3-stage 权重流水线）
 */

#include "kittens.cuh"
#include "megakernel.cuh"
#include <cuda_bf16.h>

// ============================================================================
// Opcode 定义：融合操作使用 opcode=9
// ============================================================================
#define OPCODE_FusedAttnMoE 9

// ============================================================================
// 模型配置（可通过编译宏自定义不同模型尺寸）
// ============================================================================
#ifndef FUSED_HIDDEN_DIM
#define FUSED_HIDDEN_DIM 2048          // Llama-1B hidden dimension
#endif

#ifndef FUSED_INTERMEDIATE_DIM
#define FUSED_INTERMEDIATE_DIM 8192    // MoE expert FFN 中间维度
#endif

#ifndef FUSED_HEAD_DIM
#define FUSED_HEAD_DIM 64              // 每个 attention head 的维度
#endif

#ifndef FUSED_NUM_ATTENTION_HEADS
#define FUSED_NUM_ATTENTION_HEADS 32   // Q heads 数量
#endif

#ifndef FUSED_NUM_KV_HEADS
#define FUSED_NUM_KV_HEADS 8           // KV heads 数量（GQA）
#endif

#ifndef FUSED_KV_BLOCK_SIZE
#define FUSED_KV_BLOCK_SIZE 16         // KV cache 分块大小
#endif

#ifndef FUSED_NUM_EXPERTS
#define FUSED_NUM_EXPERTS 8            // MoE 专家总数
#endif

#ifndef FUSED_NUM_EXPERTS_PER_TOK
#define FUSED_NUM_EXPERTS_PER_TOK 2    // 每个 token 选择的专家数
#endif

// ============================================================================
// Warp 分配常量
// ============================================================================
// 将 16 个 consumer warp 平分为 attention 和 MoE 两组
constexpr int FUSED_ATTN_CONSUMER_WARPS = 8;   // warp 0~7: attention
constexpr int FUSED_MOE_CONSUMER_WARPS  = 8;   // warp 8~15: MoE

// GQA 比例: 每个 KV head 对应多少 Q heads
constexpr int FUSED_GQA_RATIO = FUSED_NUM_ATTENTION_HEADS / FUSED_NUM_KV_HEADS;

// ============================================================================
// Globals 结构体
// ============================================================================
// 【设计说明】
// 复用 llama.cuh 中的 globals_t 结构，但额外添加了融合所需的字段：
// - attn_done_barrier: token 级别的 attention 完成标记
// - moe_input_activations: 已完成 attention 的 token 的激活值
// - moe_output_accumulator: MoE 输出累加缓冲区
//
// 使用 default_config（16 consumer warps），在 consumer 内部分流。
// 这样完全兼容现有的 Megakernel 框架，不需要修改 config。

template <int _num_layers, int _hidden_dim, int _intermediate_dim,
          int _head_dim, int _num_attention_heads, int _num_kv_heads,
          int _kv_block_size, int _matvec_block_size, int _sm_count,
          int _num_experts = FUSED_NUM_EXPERTS,
          int _num_experts_per_tok = FUSED_NUM_EXPERTS_PER_TOK>
struct fused_globals_t {
    constexpr static int num_layers = _num_layers;
    constexpr static int matvec_block_size = _matvec_block_size;
    constexpr static int kv_block_size = _kv_block_size;
    constexpr static int head_dim = _head_dim;
    constexpr static int hidden_dim = _hidden_dim;
    constexpr static int intermediate_dim = _intermediate_dim;
    constexpr static int num_attention_heads = _num_attention_heads;
    constexpr static int num_kv_heads = _num_kv_heads;
    constexpr static int sm_count = _sm_count;
    constexpr static int num_experts = _num_experts;
    constexpr static int num_experts_per_tok = _num_experts_per_tok;

    // 【关键】使用 default_config，不自定义 config
    // 这样与 Megakernel 框架完全兼容
    using config = megakernel::default_config;

    // VM 指令和 timing 布局
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;

    // ---- 权重类型 ----
    // QKV/O 投影权重: [num_layers, blocks, 4_col_chunks, hidden_dim]
    using weights_t = kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
                                  kittens::st_bf<matvec_block_size, 512>>;

    // KV cache: [num_layers, kv_blocks, kv_heads, kv_block_size, head_dim]
    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, -1, head_dim,
                                   kittens::sv_bf<matvec_block_size>,
                                   kittens::tma::descriptor<kittens::st_bf<kv_block_size, head_dim>, 1>>;

    // 激活值: [1, 1, 1, hidden_dim]
    using activations_t = kittens::gl<kittens::bf16, 1, 1, 1, hidden_dim,
                                      kittens::sv_bf<hidden_dim>,
                                      kittens::sv_bf<head_dim>,
                                      kittens::sv_bf<matvec_block_size>>;

    // Attention 中间结果
    using attn_out_intermediates_t = kittens::gl<float, 1, num_attention_heads, -1, head_dim,
                                                 kittens::sv_fl<head_dim>>;
    using attn_lse_intermediates_t = kittens::gl<float, 1, 1, num_attention_heads, -1,
                                                 kittens::sv_fl<((sm_count + 15) / 16) * 16>>;

    // Norm 权重
    using norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim,
                                       kittens::sv_bf<hidden_dim>,
                                       kittens::sv_bf<matvec_block_size>>;

    // RoPE 表
    using rope_table_t = kittens::gl<float, 1, 1, -1, head_dim, kittens::sv_fl<head_dim>>;

    // MoE 权重: up/gate 投影 [num_layers*num_experts, blocks, 4, hidden_dim]
    using moe_weights_t = kittens::gl<kittens::bf16, -1, -1, -1, hidden_dim,
                                      kittens::st_bf<matvec_block_size, 512>>;
    // MoE down 投影（intermediate -> hidden）
    using moe_weights_big_t = kittens::gl<kittens::bf16, -1, -1, -1, intermediate_dim,
                                          kittens::st_bf<matvec_block_size, 512>>;

    // MoE routing
    using routing_t = kittens::gl<int, 1, 1, -1, num_experts_per_tok>;
    using routing_weight_t = kittens::gl<float, 1, 1, -1, num_experts_per_tok>;

    // MoE 中间 activation
    using activations_big_indim_t = kittens::gl<kittens::bf16, 1, 1, 1, intermediate_dim,
                                                kittens::sv_bf<intermediate_dim>,
                                                kittens::sv_bf<hidden_dim>,
                                                kittens::sv_bf<matvec_block_size>>;

    // Barrier: [num_layers, num_opcodes, num_heads+2*num_kv_heads]
    using barriers = kittens::gl<uint, 1, -1, -1, num_attention_heads + 2 * num_kv_heads>;

    // ========== 数据成员 ==========

    // --- VM 基础设施 ---
    barriers Bar;
    instruction_layout instructions;
    timing_layout timings;

    // --- 模型权重 ---
    weights_t qkv_weights;
    norm_weights_t attn_norm_weights;
    weights_t o_weights;
    norm_weights_t mlp_norm_weights;

    // --- KV cache ---
    kv_cache_t k_cache;
    kv_cache_t v_cache;

    // --- RoPE ---
    rope_table_t rope_cos;
    rope_table_t rope_sin;

    // --- Activation 缓冲区 ---
    activations_t hidden_states;
    activations_t q_post_rope;
    activations_t attn_out;
    attn_lse_intermediates_t attn_lse_intermediates;
    attn_out_intermediates_t attn_out_intermediates;

    // --- MoE 权重 ---
    moe_weights_t moe_up_weights;
    moe_weights_t moe_gate_weights;
    moe_weights_big_t moe_down_weights;

    // --- MoE Routing ---
    routing_t moe_expert_indices;
    routing_weight_t moe_expert_routing_weights;

    // --- MoE 中间缓冲 ---
    activations_big_indim_t moe_intermediate;

    // ========== 融合特有字段 ==========

    // Token 级别的 attention 完成屏障
    // attn_done_barrier[token_idx] 在该 token 的 attention 完成时原子递增
    // MoE warps 通过轮询此值来判断是否可以开始 MoE 计算
    int *attn_done_barrier;  // [max_batch_size]

    // 已完成 attention 的 token 的输入激活值（post O-proj + residual）
    // MoE warps 从这里读取输入
    kittens::bf16 *moe_input_activations;  // [max_batch_size, hidden_dim]

    // MoE 输出累加器（多个 expert 的结果通过 atomicAdd 累加）
    float *moe_output_accumulator;  // [max_batch_size, hidden_dim]

    // --- 标量 ---
    unsigned int pos_id;
    float attn_scale;
    float rms_norm_eps;
    int batch_size;          // 当前 batch 中的 token 数
    int current_token_idx;   // 当前 SM 处理的 attention token 索引
    int moe_token_idx;       // MoE warps 应处理的 token 索引（-1 表示无 MoE 任务）
    bool skip_attn_reduction; // 是否跳过 attention reduction

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};
