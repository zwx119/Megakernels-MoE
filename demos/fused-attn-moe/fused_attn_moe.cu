/**
 * ============================================================================
 *   融合 Attention + MoE 算子实现
 * ============================================================================
 *
 * 【文件说明】
 *   本文件实现了融合 Attention + MoE 算子的 5 个组件：
 *   controller、loader、launcher、consumer、storer
 *
 *   这是基于 Megakernel 框架的标准算子结构。每个算子都必须实现这 5 个 struct，
 *   框架通过 MAKE_WORKER 宏生成的 main_loop + dispatch_op 来分派执行。
 *
 * 【共享内存页面布局】（总共 13 页，每页 16KB = 16384 bytes）
 *   Page 0:    Attention QOL 数据
 *              - Q: st_bf<16, 64> = 2048 bytes
 *              - O: sv_fl<64> * 4 = 1024 bytes
 *              - L: sv_fl<16> = 64 bytes
 *              总计 ≈ 3136 bytes，远小于 16KB
 *
 *   Page 1:    Attention KV cache tiles（3-stage 流水线）
 *              - 6 个 st_bf<16, 64> = 6 * 2048 = 12288 bytes < 16KB ✓
 *
 *   Pages 2~9: MoE 权重 tiles（2-stage 流水线，每 stage 4 页）
 *              - Stage 0: pages 2,3,4,5  (4 个 st_bf<16, 512> = 4*16KB)
 *              - Stage 1: pages 6,7,8,9  (4 个 st_bf<16, 512> = 4*16KB)
 *
 *   Pages 10~12: 空闲/scratch
 *
 * 【信号量布局】（总共使用 24/32 个动态信号量）
 *   [0]      Q_arrived           [1]   O_arrived        [2]   L_arrived
 *   [3..8]   K_arrived[3] + V_arrived[3]
 *   [9..14]  K_finished[3] + V_finished[3]
 *   [15]     moe_act_arrived
 *   [16..17] moe_w_arrived[2]    [18..19] moe_w_finished[2]
 *   [20..21] moe_out_arrived[2]  [22..23] moe_out_finished[2]
 */

#include "fused_attn_moe.cuh"

// 【关键】引入 utils.cuh 获取 matvec() 和 matvec_reduce() 函数
// 路径相对于当前文件，复用 low-latency-llama 中已有的 matvec 实现
#include "../low-latency-llama/utils.cuh"

using namespace kittens;
using namespace megakernel;

// 使用 default_config（16 consumer warps, 13 pages, 32 semaphores）
using config = megakernel::default_config;

// ============================================================================
// 类型别名 - Attention 部分
// ============================================================================
using attn_q_rt  = kittens::rt_bf<16, FUSED_HEAD_DIM>;
using attn_q_st  = kittens::st_bf<16, FUSED_HEAD_DIM>;
using attn_k_rt  = kittens::rt_bf<FUSED_KV_BLOCK_SIZE, FUSED_HEAD_DIM>;
using attn_v_rt  = kittens::rt_bf<FUSED_KV_BLOCK_SIZE, FUSED_HEAD_DIM, kittens::col_l>;
using attn_kv_st = kittens::st_bf<FUSED_KV_BLOCK_SIZE, FUSED_HEAD_DIM>;
using attn_fl_rt = kittens::rt_fl<16, FUSED_KV_BLOCK_SIZE>;
using attn_bf_rt = kittens::rt_bf<16, FUSED_KV_BLOCK_SIZE>;
using attn_o_rt  = kittens::rt_fl<16, FUSED_HEAD_DIM>;
using attn_o_sv  = kittens::sv_fl<FUSED_HEAD_DIM>;
using attn_o_sv_bf = kittens::sv_bf<FUSED_HEAD_DIM>;
using max_vec_rv  = col_vec<kittens::rt_fl<16, FUSED_HEAD_DIM>>;
using norm_vec_rv = col_vec<kittens::rt_fl<16, FUSED_HEAD_DIM>>;
using l_rv = col_vec<kittens::rt_fl<16, FUSED_HEAD_DIM>>;
using l_sv = kittens::sv_fl<16>;

// ============================================================================
// 常量定义
// ============================================================================
// 信号量索引
constexpr int SEM_ATTN_Q_ARRIVED  = 0;
constexpr int SEM_ATTN_O_ARRIVED  = 1;
constexpr int SEM_ATTN_L_ARRIVED  = 2;
constexpr int SEM_ATTN_KV_BASE    = 3;
constexpr int ATTN_NUM_STAGES     = 3;
constexpr int SEM_MOE_ACT_ARRIVED      = 15;
constexpr int SEM_MOE_W_ARRIVED_BASE   = 16;
constexpr int SEM_MOE_W_FINISHED_BASE  = 18;
constexpr int SEM_MOE_OUT_ARRIVED_BASE = 20;
constexpr int SEM_MOE_OUT_FINISHED_BASE = 22;

// MoE 流水线和页面
constexpr int MOE_PIPELINE_STAGES = 2;
constexpr int PAGE_ATTN_QOL    = 0;
constexpr int PAGE_ATTN_KV     = 1;
constexpr int PAGE_MOE_W_START = 2;
constexpr int MOE_STAGE_PAGES  = 4;

// MoE 计算参数
constexpr int MOE_REDUCTION_DIM_PER_WARP = FUSED_HIDDEN_DIM / FUSED_MOE_CONSUMER_WARPS;
constexpr int MOE_SCRATCH_PER_WARP = 16 * sizeof(float);

// ============================================================================
// 共享内存访问器
// ============================================================================

template <typename cfg>
struct fused_smem {
    __device__ static inline attn_q_st &get_Q(megakernel::state<cfg> &s) {
        return *reinterpret_cast<attn_q_st *>(s.pages[s.pid(PAGE_ATTN_QOL)].data);
    }

    __device__ static inline attn_o_sv (&get_O(megakernel::state<cfg> &s))[FUSED_GQA_RATIO] {
        int pid = s.pid(PAGE_ATTN_QOL);
        return *reinterpret_cast<attn_o_sv(*)[FUSED_GQA_RATIO]>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(attn_q_st));
    }

    __device__ static inline l_sv &get_L(megakernel::state<cfg> &s) {
        int pid = s.pid(PAGE_ATTN_QOL);
        return *reinterpret_cast<l_sv *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(attn_q_st) +
            sizeof(attn_o_sv) * FUSED_GQA_RATIO);
    }

    __device__ static inline attn_kv_st &get_K(megakernel::state<cfg> &s, int stage) {
        int pid = s.pid(PAGE_ATTN_KV);
        return *reinterpret_cast<attn_kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(attn_kv_st) * (stage * 2));
    }

    __device__ static inline attn_kv_st &get_V(megakernel::state<cfg> &s, int stage) {
        int pid = s.pid(PAGE_ATTN_KV);
        return *reinterpret_cast<attn_kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(attn_kv_st) * (stage * 2 + 1));
    }

    __device__ static inline kittens::st_bf<16, 512> &get_moe_weight(
            megakernel::state<cfg> &s, int stage, int col_chunk) {
        int pid = s.pid(PAGE_MOE_W_START + stage * MOE_STAGE_PAGES + col_chunk);
        return reinterpret_cast<kittens::st_bf<16, 512> &>(s.pages[pid]);
    }
};

// ============================================================================
// 信号量访问器
// ============================================================================

template <typename cfg>
struct fused_sems {
    __device__ static inline kittens::semaphore &Q_arrived(megakernel::state<cfg> &s) {
        return s.semaphores()[SEM_ATTN_Q_ARRIVED];
    }
    __device__ static inline kittens::semaphore &O_arrived(megakernel::state<cfg> &s) {
        return s.semaphores()[SEM_ATTN_O_ARRIVED];
    }
    __device__ static inline kittens::semaphore &L_arrived(megakernel::state<cfg> &s) {
        return s.semaphores()[SEM_ATTN_L_ARRIVED];
    }
    __device__ static inline kittens::semaphore &K_arrived(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_ATTN_KV_BASE + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_arrived(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_ATTN_KV_BASE + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &K_finished(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_ATTN_KV_BASE + ATTN_NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_finished(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_ATTN_KV_BASE + ATTN_NUM_STAGES * 2 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &moe_act_arrived(megakernel::state<cfg> &s) {
        return s.semaphores()[SEM_MOE_ACT_ARRIVED];
    }
    __device__ static inline kittens::semaphore &moe_w_arrived(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_MOE_W_ARRIVED_BASE + stage];
    }
    __device__ static inline kittens::semaphore &moe_w_finished(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_MOE_W_FINISHED_BASE + stage];
    }
    __device__ static inline kittens::semaphore &moe_out_arrived(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_MOE_OUT_ARRIVED_BASE + stage];
    }
    __device__ static inline kittens::semaphore &moe_out_finished(megakernel::state<cfg> &s, int stage) {
        return s.semaphores()[SEM_MOE_OUT_FINISHED_BASE + stage];
    }
};

// ============================================================================
// 指令解析
// ============================================================================

struct fused_parsed_instruction {
    int layer_idx;
    int kv_head_idx;
    int num_partials;
    int partial_idx;
    int moe_token_idx;
    int moe_expert_idx;
    int moe_weight_type;
    int moe_start_block;
    int moe_end_block;
    int moe_reduction_block;

    __device__ inline int moe_iters() const { return moe_end_block - moe_start_block; }
    __device__ inline bool has_moe_work() const { return moe_token_idx >= 0; }

    template <typename cfg>
    __device__ inline fused_parsed_instruction(megakernel::state<cfg> &s) {
        auto &inst = s.instruction();
        layer_idx          = inst[1];
        kv_head_idx        = inst[2];
        num_partials       = inst[3];
        partial_idx        = inst[4];
        moe_token_idx      = inst[5];
        moe_expert_idx     = inst[6];
        moe_weight_type    = inst[7];
        moe_start_block    = inst[8];
        moe_end_block      = inst[9];
        moe_reduction_block = inst[10];
    }
};

// ============================================================================
// 融合 Attention + MoE 算子
// ============================================================================

template <typename Config, typename Globals>
struct fused_attn_moe_op {
    static constexpr int opcode = OPCODE_FusedAttnMoE;

    using smem_t = fused_smem<Config>;
    using sems_t = fused_sems<Config>;
    using inst_t = fused_parsed_instruction;

    // ========================================================================
    // Controller
    // ========================================================================
    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            // 释放顺序: 空闲页 → attn 页 → MoE 权重页
            int ret_order[13] = {10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            return ret_order[query];
        }

        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            // Attention 信号量
            init_semaphore(sems_t::Q_arrived(s), 0, 1);
            init_semaphore(sems_t::O_arrived(s), 0, 1);
            init_semaphore(sems_t::L_arrived(s), 0, 1);
            for (int i = 0; i < ATTN_NUM_STAGES; i++) {
                init_semaphore(sems_t::K_arrived(s, i), 0, 1);
                init_semaphore(sems_t::V_arrived(s, i), 0, 1);
                init_semaphore(sems_t::K_finished(s, i), 0, 1);
                init_semaphore(sems_t::V_finished(s, i), 0, 1);
            }
            // MoE 信号量
            init_semaphore(sems_t::moe_act_arrived(s), 0, 1);
            for (int i = 0; i < MOE_PIPELINE_STAGES; i++) {
                init_semaphore(sems_t::moe_w_arrived(s, i), 0, 1);
                init_semaphore(sems_t::moe_w_finished(s, i), 0, FUSED_MOE_CONSUMER_WARPS);
                init_semaphore(sems_t::moe_out_arrived(s, i), 0, FUSED_MOE_CONSUMER_WARPS);
                init_semaphore(sems_t::moe_out_finished(s, i), 0, 1);
            }
            return 24;
        }
    };

    // ========================================================================
    // Loader
    // ========================================================================
    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            inst_t inst{s};
            auto laneid = kittens::warp::laneid();

            if (laneid == 0) {
                load_attn_kv(g, s, inst);
                if (inst.has_moe_work()) {
                    load_moe_weights(g, s, inst);
                }
            }

            // 释放未使用的页面
            int moe_pages = inst.has_moe_work() ?
                min(inst.moe_iters(), MOE_PIPELINE_STAGES) * MOE_STAGE_PAGES : 0;
            int needed_pages = 2 + moe_pages;
            if (laneid >= needed_pages && laneid < Config::NUM_PAGES) {
                auto pid = s.pid(laneid);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }

    private:
        static __device__ void load_attn_kv(const Globals &g,
                                             megakernel::state<Config> &s,
                                             inst_t &inst) {
            int seq_len = g.pos_id + 1;
            int total_blocks = (seq_len + FUSED_KV_BLOCK_SIZE - 1) / FUSED_KV_BLOCK_SIZE;
            int blocks_per_partial = (total_blocks + inst.num_partials - 1) / inst.num_partials;
            int start_blk = inst.partial_idx * blocks_per_partial;
            int end_blk = min(start_blk + blocks_per_partial, total_blocks);

            s.wait_page_ready(s.pid(PAGE_ATTN_QOL));
            s.wait_page_ready(s.pid(PAGE_ATTN_KV));

            if (start_blk >= end_blk) {
                s.finish_page(s.pid(PAGE_ATTN_KV), Config::NUM_CONSUMER_WARPS);
                return;
            }

            for (int i = 0; i + start_blk < end_blk; ++i) {
                int cur_blk = start_blk + i;
                int stage = i % ATTN_NUM_STAGES;
                attn_kv_st &K_smem = smem_t::get_K(s, stage);
                attn_kv_st &V_smem = smem_t::get_V(s, stage);

                if (i >= ATTN_NUM_STAGES) {
                    kittens::wait(sems_t::K_finished(s, stage), (i / ATTN_NUM_STAGES - 1) % 2);
                    kittens::wait(sems_t::V_finished(s, stage), (i / ATTN_NUM_STAGES - 1) % 2);
                }

                kittens::tma::expect(sems_t::K_arrived(s, stage), K_smem);
                kittens::tma::load_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                    K_smem, g.k_cache,
                    {inst.layer_idx, cur_blk, inst.kv_head_idx, 0},
                    sems_t::K_arrived(s, stage));

                kittens::tma::expect(sems_t::V_arrived(s, stage), V_smem);
                kittens::tma::load_async<kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                    V_smem, g.v_cache,
                    {inst.layer_idx, cur_blk, inst.kv_head_idx, 0},
                    sems_t::V_arrived(s, stage));
            }
        }

        static __device__ void load_moe_weights(const Globals &g,
                                                  megakernel::state<Config> &s,
                                                  inst_t &inst) {
            int num_iters = inst.moe_iters();
            if (num_iters <= 0) return;

            int actual_expert_id = g.moe_expert_indices.raw_ptr[
                inst.layer_idx * Globals::num_experts_per_tok + inst.moe_expert_idx
            ];
            int row_idx = inst.layer_idx * Globals::num_experts + actual_expert_id;

            // 等待初始 MoE 页面就绪
            for (int stage = 0; stage < min(num_iters, MOE_PIPELINE_STAGES); stage++) {
                for (int col = 0; col < MOE_STAGE_PAGES; col++) {
                    s.wait_page_ready(s.pid(PAGE_MOE_W_START + stage * MOE_STAGE_PAGES + col));
                }
            }

            kittens::arrive(sems_t::moe_act_arrived(s));

            int input_stage = 0;
            for (int iter = 0; iter < num_iters; iter++) {
                if (iter >= MOE_PIPELINE_STAGES) {
                    kittens::wait(sems_t::moe_w_finished(s, input_stage),
                                  (iter / MOE_PIPELINE_STAGES - 1) % 2);
                }

                auto &sem = sems_t::moe_w_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * 2048 * 16);

                int block_idx = inst.moe_start_block + iter;
                for (int col = 0; col < MOE_STAGE_PAGES; col++) {
                    auto &weight_chunk = smem_t::get_moe_weight(s, input_stage, col);

                    switch (inst.moe_weight_type) {
                        case 0:
                            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                                weight_chunk, g.moe_up_weights,
                                {row_idx, block_idx, col}, sem);
                            break;
                        case 1:
                            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                                weight_chunk, g.moe_gate_weights,
                                {row_idx, block_idx, col}, sem);
                            break;
                        case 2:
                            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                                weight_chunk, g.moe_down_weights,
                                {row_idx, block_idx,
                                 inst.moe_reduction_block * Globals::hidden_dim + col * 512},
                                sem);
                            break;
                    }
                }

                input_stage = (input_stage + 1) % MOE_PIPELINE_STAGES;
            }
        }
    };

    // ========================================================================
    // Launcher
    // ========================================================================
    struct launcher {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            if (kittens::warp::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };

    // ========================================================================
    // Consumer: 融合执行的核心
    // ========================================================================
    // 【设计】
    //   warpid < 8  → attention_consumer (只 warp 0 工作)
    //   warpid >= 8 → moe_consumer (所有 8 个 warp 协作)
    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            int warp_id = kittens::warpid();

            if (warp_id < FUSED_ATTN_CONSUMER_WARPS) {
                if (warp_id == 0) {
                    attention_consumer(g, s);
                }
                // warp 1~7: 空闲（可扩展处理更多 Q heads）
            } else {
                inst_t inst{s};
                if (inst.has_moe_work()) {
                    moe_consumer(g, s, warp_id - FUSED_ATTN_CONSUMER_WARPS, inst);
                }
            }
        }

    private:
        // ---- Attention consumer ----
        static __device__ void attention_consumer(const Globals &g,
                                                   megakernel::state<Config> &s) {
            inst_t inst{s};
            int q_head_start_idx = inst.kv_head_idx * FUSED_GQA_RATIO;

            // 等待 QKV 计算完成
            if (kittens::laneid() == 0) {
                for (int head_offset = 0; head_offset < FUSED_GQA_RATIO; head_offset++) {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, 0,
                            q_head_start_idx + head_offset}] < 4) {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                }
            }
            kittens::warp::sync();

            // 计算参数
            int seq_len = g.pos_id + 1;
            int total_blocks = (seq_len + FUSED_KV_BLOCK_SIZE - 1) / FUSED_KV_BLOCK_SIZE;
            int blocks_per_partial = (total_blocks + inst.num_partials - 1) / inst.num_partials;
            int start_blk = inst.partial_idx * blocks_per_partial;
            int end_blk = min(start_blk + blocks_per_partial, total_blocks);
            float softmax_temp = g.attn_scale * 1.44269504089f;

            // 寄存器分配
            attn_q_rt Q_reg;
            attn_k_rt K_reg;
            attn_v_rt V_reg;
            l_rv L_reg;
            attn_o_rt O_reg;
            attn_fl_rt attn_fl_reg;
            attn_bf_rt attn_bf_reg;
            max_vec_rv max_vec_reg, scaled_max_vec_reg;
            max_vec_rv last_scaled_max_vec_reg, diff_scaled_max_vec_reg;
            norm_vec_rv norm_vec_reg;

            kittens::warp::neg_infty(max_vec_reg);
            kittens::warp::zero(last_scaled_max_vec_reg);
            kittens::warp::zero(norm_vec_reg);
            kittens::warp::zero(O_reg);

            attn_o_sv (&O_smem)[FUSED_GQA_RATIO] = smem_t::get_O(s);
            l_sv &L_smem = smem_t::get_L(s);

            // 加载 Q
            s.wait_page_ready(s.pid(PAGE_ATTN_QOL));
            attn_q_st &Q_smem = smem_t::get_Q(s);

            {
                using T = typename attn_q_st::dtype;
                constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
                constexpr int memcpy_per_row = FUSED_HEAD_DIM / elem_per_memcpy;
                auto *src_ptr = &g.q_post_rope.raw_ptr[q_head_start_idx * FUSED_HEAD_DIM];
                uint32_t dst_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&Q_smem.data[(q_head_start_idx % 16) * FUSED_HEAD_DIM]));
                int lane = kittens::warp::laneid();
                int row = lane / memcpy_per_row;
                int col = (lane * elem_per_memcpy) % FUSED_HEAD_DIM;
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    ::"r"(Q_smem.idx(dst_ptr, {row, col})),
                      "l"(&src_ptr[row * FUSED_HEAD_DIM + col])
                    : "memory");
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }

            kittens::warp::load_async_wait();
            kittens::warp::load(Q_reg, Q_smem);

            // Flash Attention 主循环
            for (int i = 0; i + start_blk < end_blk; ++i) {
                int stage = i % ATTN_NUM_STAGES;
                attn_kv_st &K_smem = smem_t::get_K(s, stage);
                attn_kv_st &V_smem = smem_t::get_V(s, stage);

                kittens::warp::zero(attn_fl_reg);
                kittens::warp::wait(sems_t::K_arrived(s, stage), (i / ATTN_NUM_STAGES) % 2);
                kittens::warp::load(K_reg, K_smem);
                kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                kittens::warp::sync();
                kittens::warp::arrive(sems_t::K_finished(s, stage));

                // Masking
                if ((i + start_blk + 1) * FUSED_KV_BLOCK_SIZE > seq_len) {
                    int valid_cols = seq_len % FUSED_KV_BLOCK_SIZE;
                    #pragma unroll
                    for (int r = 0; r < attn_fl_reg.height; r++) {
                        #pragma unroll
                        for (int c = 0; c < attn_fl_reg.width; c++) {
                            #pragma unroll
                            for (int k = 0; k < attn_fl_reg.packed_per_tile; k++) {
                                int col_x = c * attn_fl_reg.tile_size_col +
                                           (k / 2) * 8 + (kittens::warp::laneid() % 4) * 2;
                                int col_y = col_x + 1;
                                if (col_x >= valid_cols)
                                    attn_fl_reg.tiles[r][c].data[k].x = -999999999999.f;
                                if (col_y >= valid_cols)
                                    attn_fl_reg.tiles[r][c].data[k].y = -999999999999.f;
                            }
                        }
                    }
                }

                // Online softmax
                kittens::warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);
                kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                kittens::warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);
                kittens::warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                kittens::warp::exp2(attn_fl_reg, attn_fl_reg);
                kittens::warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                kittens::warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                kittens::warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                kittens::warp::wait(sems_t::V_arrived(s, stage), (i / ATTN_NUM_STAGES) % 2);
                kittens::warp::load(V_reg, V_smem);
                kittens::warp::copy(attn_bf_reg, attn_fl_reg);
                kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                kittens::warp::sync();
                kittens::warp::arrive(sems_t::V_finished(s, stage));

                kittens::warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);
                kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
            }

            // 最终化
            kittens::warp::sync();

            if (start_blk < end_blk) {
                if (kittens::warp::laneid() == 0) {
                    s.finish_page(s.pid(PAGE_ATTN_KV), Config::NUM_CONSUMER_WARPS);
                }
                kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);
                kittens::warp::log2(L_reg, norm_vec_reg);
                kittens::warp::add(L_reg, L_reg, last_scaled_max_vec_reg);
            } else {
                kittens::warp::neg_infty(L_reg);
            }

            for (int h = 0; h < FUSED_GQA_RATIO; h++) {
                kittens::warp::store(O_smem[h], O_reg);
            }
            kittens::warp::sync();
            kittens::warp::arrive(sems_t::O_arrived(s));

            kittens::warp::store(L_smem, L_reg);
            kittens::warp::sync();
            kittens::warp::arrive(sems_t::L_arrived(s));

            // 标记 attention 完成
            if (kittens::laneid() == 0 && g.attn_done_barrier != nullptr) {
                atomicAdd(&g.attn_done_barrier[g.current_token_idx], 1);
            }
        }

        // ---- MoE consumer ----
        static __device__ void moe_consumer(const Globals &g,
                                             megakernel::state<Config> &s,
                                             int local_moe_warp_id,
                                             inst_t &inst) {
            // 等待目标 token 的 attention 完成
            if (local_moe_warp_id == 0 && kittens::laneid() == 0) {
                while (g.attn_done_barrier != nullptr &&
                       *(volatile int *)&g.attn_done_barrier[inst.moe_token_idx] < 1) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
            }
            __syncwarp();

            // 等待 loader 信号
            kittens::wait(sems_t::moe_act_arrived(s), 0);

            // 每个 warp 加载自己负责的 activation 片段
            constexpr int RED_DIM = MOE_REDUCTION_DIM_PER_WARP;
            using moe_rv_t = kittens::rv_fl<RED_DIM>;
            moe_rv_t activations_vec;

            {
                kittens::bf16 *src;
                if (inst.moe_weight_type == 2) {
                    int offset = inst.moe_reduction_block * FUSED_HIDDEN_DIM +
                                 local_moe_warp_id * RED_DIM;
                    src = &g.moe_intermediate.raw_ptr[offset];
                } else {
                    int offset = inst.moe_token_idx * FUSED_HIDDEN_DIM +
                                 local_moe_warp_id * RED_DIM;
                    src = g.moe_input_activations + offset;
                }

                // 加载到共享内存临时区域，再加载到寄存器
                // 使用 pages[10] 作为临时 activation 缓冲
                int act_pid = s.pid(PAGE_MOE_W_START + MOE_PIPELINE_STAGES * MOE_STAGE_PAGES);
                using act_sv_t = kittens::sv_bf<RED_DIM>;
                act_sv_t &act_smem = reinterpret_cast<act_sv_t *>(
                    s.pages[act_pid].ptr())[local_moe_warp_id];

                kittens::warp::load(act_smem, *reinterpret_cast<kittens::gl<kittens::bf16, 1, 1, 1, RED_DIM> *>(src), {});

                // 转换 bf16 → fp32
                kittens::warp::load(activations_vec, act_smem);
                kittens::warp::sync();
            }

            // MoE matvec 流水线
            int num_iters = inst.moe_iters();
            constexpr int WARPS_PER_PAGE = FUSED_MOE_CONSUMER_WARPS / MOE_STAGE_PAGES;
            int page_index = local_moe_warp_id / WARPS_PER_PAGE;

            int input_stage = 0, output_stage = 0;

            for (int i = 0; i < num_iters; i++) {
                kittens::wait(sems_t::moe_w_arrived(s, input_stage),
                              (i % (2 * MOE_PIPELINE_STAGES)) >= MOE_PIPELINE_STAGES);
                kittens::wait(sems_t::moe_out_finished(s, output_stage),
                              (i % (2 * MOE_PIPELINE_STAGES)) < MOE_PIPELINE_STAGES);

                int weight_page = s.pid(PAGE_MOE_W_START + input_stage * MOE_STAGE_PAGES + page_index);
                kittens::st_bf<16, RED_DIM> &weights =
                    reinterpret_cast<kittens::st_bf<16, RED_DIM> *>(
                        s.pages[weight_page].ptr())[local_moe_warp_id % WARPS_PER_PAGE];

                uint8_t *output_scratch = (uint8_t *)s.scratch() +
                    (output_stage * FUSED_MOE_CONSUMER_WARPS + local_moe_warp_id) * MOE_SCRATCH_PER_WARP;
                kittens::sv_fl<16> &out_smem = *reinterpret_cast<kittens::sv_fl<16> *>(output_scratch);

                matvec(out_smem, weights, activations_vec);

                kittens::warp::sync();
                kittens::warp::arrive(sems_t::moe_out_arrived(s, output_stage));
                kittens::warp::arrive(sems_t::moe_w_finished(s, input_stage));

                if (i >= num_iters - MOE_PIPELINE_STAGES) {
                    for (int j = 0; j < MOE_STAGE_PAGES; j++) {
                        s.warp_finish_page(
                            s.pid(PAGE_MOE_W_START + input_stage * MOE_STAGE_PAGES + j), 1);
                    }
                }

                input_stage = (input_stage + 1) % MOE_PIPELINE_STAGES;
                output_stage = (output_stage + 1) % MOE_PIPELINE_STAGES;
            }
        }
    };

    // ========================================================================
    // Storer
    // ========================================================================
    struct storer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            inst_t inst{s};
            int laneid = kittens::warp::laneid();

            store_attention_output(g, s, inst, laneid);

            if (inst.has_moe_work()) {
                store_moe_output(g, s, inst, laneid);
            }
        }

    private:
        static __device__ void store_attention_output(const Globals &g,
                                                       megakernel::state<Config> &s,
                                                       inst_t &inst,
                                                       int laneid) {
            int q_head_start_idx = inst.kv_head_idx * FUSED_GQA_RATIO;
            auto O_smem = smem_t::get_O(s);

            if (laneid == 0) {
                kittens::wait(sems_t::O_arrived(s), 0);
                s.record(megakernel::TEVENT_OUTPUT_READY);
            }
            kittens::warp::sync();

            if (g.skip_attn_reduction) {
                kittens::rv_bf<FUSED_HEAD_DIM> O_bf;
                for (int head_offset = 0; head_offset < FUSED_GQA_RATIO; head_offset++) {
                    auto &smem_fl = O_smem[head_offset];
                    auto &smem_bf = *reinterpret_cast<attn_o_sv_bf *>(&smem_fl);
                    kittens::warp::load(O_bf, smem_fl);
                    kittens::warp::sync();
                    kittens::warp::store(smem_bf, O_bf);
                    kittens::warp::sync();
                }

                if (laneid == 0) {
                    for (int head_offset = 0; head_offset < FUSED_GQA_RATIO; head_offset++) {
                        auto &smem_bf = *reinterpret_cast<attn_o_sv_bf *>(&O_smem[head_offset]);
                        kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                            g.attn_out, smem_bf, {q_head_start_idx + head_offset});
                    }
                }
            } else {
                if (laneid == 0) {
                    for (int head_offset = 0; head_offset < FUSED_GQA_RATIO; head_offset++) {
                        kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                            g.attn_out_intermediates, O_smem[head_offset],
                            {0, q_head_start_idx + head_offset, inst.partial_idx, 0});
                    }
                }
            }

            if (laneid < FUSED_GQA_RATIO && !g.skip_attn_reduction) {
                l_sv &L_smem = smem_t::get_L(s);
                kittens::wait(sems_t::L_arrived(s), 0);

                float tmp;
                uint32_t src_ptr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&L_smem.data[(q_head_start_idx % 16) + laneid]));
                float *dst_ptr = (float *)&g.attn_lse_intermediates.raw_ptr[
                    (q_head_start_idx + laneid) * g.attn_lse_intermediates.cols() +
                    inst.partial_idx];
                asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(tmp) : "r"(src_ptr));
                asm volatile("st.global.f32 [%0], %1;\n" :: "l"(dst_ptr), "f"(tmp));
            }

            kittens::warp::sync();
            kittens::tma::store_async_wait();

            if (laneid == 0) {
                s.finish_page(s.pid(PAGE_ATTN_QOL), Config::NUM_CONSUMER_WARPS);
            }

            if (laneid < FUSED_GQA_RATIO) {
                if (g.skip_attn_reduction) {
                    atomicAdd(&g.Bar[{inst.layer_idx, OPCODE_FusedAttnMoE - 1, 0}], 1);
                } else {
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1,
                                      q_head_start_idx + laneid}], 1);
                }
            }
        }

        static __device__ void store_moe_output(const Globals &g,
                                                  megakernel::state<Config> &s,
                                                  inst_t &inst,
                                                  int laneid) {
            int num_iters = inst.moe_iters();
            int output_stage = 0;

            for (int i = 0; i < num_iters; i++) {
                auto &sem = sems_t::moe_out_arrived(s, output_stage);
                auto bit = (i % (2 * MOE_PIPELINE_STAGES)) >= MOE_PIPELINE_STAGES;
                kittens::wait(sem, bit);

                uint8_t *output_scratch = (uint8_t *)s.scratch() +
                    output_stage * FUSED_MOE_CONSUMER_WARPS * MOE_SCRATCH_PER_WARP;

                // 跨 warp 归约（注意这里用的是 FUSED_MOE_CONSUMER_WARPS 而非 NUM_CONSUMER_WARPS）
                kittens::rv_fl<16> result;
                kittens::warp::zero(result);
                for (int w = 0; w < FUSED_MOE_CONSUMER_WARPS; w++) {
                    kittens::sv_fl<16> &partial = *reinterpret_cast<kittens::sv_fl<16> *>(
                        output_scratch + w * MOE_SCRATCH_PER_WARP);
                    kittens::rv_fl<16> partial_vec;
                    kittens::warp::load(partial_vec, partial);
                    kittens::warp::add(result, result, partial_vec);
                }

                int block_idx = inst.moe_start_block + i;
                kittens::sv_bf<16> &out_bf = *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);

                if (inst.moe_weight_type == 2) {
                    float routing_weight = g.moe_expert_routing_weights.raw_ptr[
                        inst.layer_idx * Globals::num_experts_per_tok + inst.moe_expert_idx
                    ];
                    kittens::warp::mul(result, result, routing_weight);

                    kittens::warp::store(out_bf, result);
                    kittens::warp::sync();

                    if (laneid == 0) {
                        float *out_ptr = g.moe_output_accumulator +
                                          inst.moe_token_idx * FUSED_HIDDEN_DIM +
                                          block_idx * 16;
                        for (int j = 0; j < 16; j++) {
                            atomicAdd(&out_ptr[j], __bfloat162float(out_bf.data[j]));
                        }
                        atomicAdd((unsigned int *)&g.Bar[{inst.layer_idx, opcode - 1, 0}], 1);
                    }
                } else {
                    kittens::warp::store(out_bf, result);
                    kittens::warp::sync();

                    if (laneid == 0) {
                        kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                            g.moe_intermediate, out_bf, {block_idx});
                        kittens::tma::store_async_read_wait();
                    }
                }

                kittens::warp::arrive(sems_t::moe_out_finished(s, output_stage));
                output_stage = (output_stage + 1) % MOE_PIPELINE_STAGES;
            }
        }
    };
};
