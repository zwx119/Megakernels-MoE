#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

using globals = llama_1b_globals;
using config = default_config;

/**
 * MoE Expert MatVec Operator
 * 
 * Reuses the matvec_pipeline infrastructure.
 * 
 * Instruction format (32 ints):
 *   [0] opcode = 8
 *   [1] layer_idx
 *   [2] expert_idx (index in top-k)
 *   [3] weight_type (0=up_proj, 1=gate_proj, 2=down_proj)
 *   [4] start_block_idx
 *   [5] end_block_idx
 *   [6] reduction_block_idx
 */
template <typename Config, typename Globals> struct moe_expert_op {
    static constexpr int opcode = OPCODE_MoEExpertMatVec;
    static constexpr int prev_opcode = OPCODE_O_ProjResidual;
    static constexpr int EXPECTED_ARRIVAL_COUNT =
        Globals::hidden_dim / Globals::matvec_block_size;

    struct parsed_instruction {
        int layer_idx, expert_idx, weight_type;
        int start_block_idx, end_block_idx, reduction_block_idx;
        int iters;

        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            expert_idx = instruction[2];
            weight_type = instruction[3];
            start_block_idx = instruction[4];
            end_block_idx = instruction[5];
            reduction_block_idx = instruction[6];
            
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        /**
         * Wait for O_Proj to finish.
         */
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            while (
                *(volatile int *)&g.Bar[{inst.layer_idx, prev_opcode - 1, 0}] <
                EXPECTED_ARRIVAL_COUNT) {
                __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
            }
        }

        /**
         * Load weight tiles.
         */
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            
            // Get actual expert ID from routing indices
            int actual_expert_id = 
                g.moe_expert_indices.raw_ptr[
                    inst.layer_idx * Globals::num_experts_per_tok + inst.expert_idx
                ];
            
            int block_idx = inst.start_block_idx + iter;
            int row_idx = inst.layer_idx * Globals::num_experts + actual_expert_id;
            
            switch (inst.weight_type) {
                case 0: // up_proj
                    kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        weight_chunk, g.moe_up_weights,
                        {row_idx, block_idx, col_idx}, sem);
                    break;
                case 1: // gate_proj
                    kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        weight_chunk, g.moe_gate_weights,
                        {row_idx, block_idx, col_idx}, sem);
                    break;
                case 2: // down_proj
                    // For down_proj, the dimensions are flipped (intermediate -> hidden)
                    // The weights_big_indim_t is used.
                    kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        weight_chunk, g.moe_down_weights,
                        coord<>{row_idx, block_idx, inst.reduction_block_idx * Globals::hidden_dim + col_idx * 512}, sem);
                    break;
            }
        }

        /**
         * Store results and handle SiLU/Scaling.
         */
        static __device__ inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;
            
            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            
            kittens::rv_fl<16> result;
            
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, result);
            
            kittens::sv_bf<16> &out_smem =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);
            
            switch (inst.weight_type) {
                case 0: { // up_proj -> write to moe_intermediate
                    // Store 'up' result to moe_intermediate for later use by gate_proj
                    kittens::warp::store(out_smem, result);
                    kittens::warp::sync();
                    if (kittens::laneid() == 0) {
                        kittens::tma::store_async<cache_policy::EVICT_LAST>(
                            g.moe_intermediate, out_smem, {block_idx});
                        kittens::tma::store_async_read_wait();
                    }
                    break;
                }
                case 1: { // gate_proj -> SiLU(gate) * up, write to moe_intermediate
                    kittens::rv_fl<16> gate_scratch;
                    // SiLU(gate) = gate * sigmoid(gate)
                    kittens::warp::mul(gate_scratch, result, -1.f);
                    kittens::warp::exp(gate_scratch, gate_scratch);
                    kittens::warp::add(gate_scratch, gate_scratch, 1.f);
                    kittens::warp::div(gate_scratch, result, gate_scratch); // gate_scratch = silu(result)
                    
                    // Optimization: We know that up_proj (weight_type 0) for the same block
                    // was just executed by this same SM in the same sequence of instructions
                    // (because of how scheduler assigns up/gate pairs).
                    // So we can try to reuse shared memory if the buffer persists.
                    // However, 'output_scratch_start' is reused for the current stage.
                    // But maybe we can read from global memory which should hit L2.
                    
                    // Load 'up' result
                    kittens::sv_bf<16> up_smem;
                    kittens::warp::load(up_smem, g.moe_intermediate, {block_idx});
                    kittens::warp::sync();
                    
                    kittens::rv_fl<16> up_val;
                    kittens::warp::load(up_val, up_smem);
                    
                    // result = silu(gate) * up
                    kittens::warp::mul(result, gate_scratch, up_val);
                    
                    kittens::warp::store(out_smem, result);
                    kittens::warp::sync();
                    
                    if (kittens::laneid() == 0) {
                        kittens::tma::store_async<cache_policy::EVICT_LAST>(
                            g.moe_intermediate, out_smem, {block_idx});
                        kittens::tma::store_async_read_wait();
                    }
                    break;
                }
                case 2: { // down_proj -> add to hidden_states with routing weight
                    float routing_weight = 
                        g.moe_expert_routing_weights.raw_ptr[
                            inst.layer_idx * Globals::num_experts_per_tok + inst.expert_idx
                        ];
                    
                    kittens::warp::mul(result, result, routing_weight);
                    
                    kittens::warp::store(out_smem, result);
                    kittens::warp::sync();
                    
                    if (kittens::laneid() == 0) {
                        // Atomic add to hidden_states (multiple experts contribute)
                        float* hidden_ptr = (float*)g.hidden_states.raw_ptr;
                        int base_idx = block_idx * 16; 
                        
                        // We must cast bf16 to float for atomicAdd
                        for(int i = 0; i < 16; i++) {
                             float val = __bfloat162float(out_smem.data[i]);
                             atomicAdd(&hidden_ptr[base_idx + i], val);
                        }
                        
                         atomicAdd((unsigned int*)&g.Bar[{inst.layer_idx, opcode - 1, 0}], 1);
                    }
                    break;
                }
            }
            
            kittens::warp::sync();
        }
    };

    // Re-use rms_matvec_pipeline for MoE Expert
    using pipeline =
        rms_matvec_pipeline<Config, Globals, parsed_instruction,
                            pipeline_specifics, &Globals::hidden_states,
                            &Globals::mlp_norm_weights>;

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            // Wait for O_Proj explicitly in the first consumer if needed, 
            // or rely on pipeline::consumer_loop's internal synchronization.
            // matvec_pipeline usually handles it via gmem_wait in loader? No, specifics::gmem_wait.
            
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                pipeline_specifics::gmem_wait(g, s);
            }
            __syncthreads();

            // Load activations from hidden_states (post-LN/residual) or moe_intermediate
            parsed_instruction inst{s};
            
            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            
            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            if (inst.weight_type == 2) {
                // down_proj loads from moe_intermediate
                kittens::warp::load(activations_smem, g.moe_intermediate,
                           coord<>{inst.reduction_block_idx * Globals::hidden_dim +
                                   kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            } else {
                // up/gate load from hidden_states
                kittens::warp::load(activations_smem, g.hidden_states,
                           coord<>{kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            }
            kittens::warp::sync();

            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            // Tell loader we are ready (decrement activation semaphore)
            s.warp_finish_page(pipeline::get_activation_page(s), 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
        }
    };
};
