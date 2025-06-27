/*******************************************************************************
* Copyright (c) 2022-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "xetla_lora_a.h"
#include "xetla_lora_b.h"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n_A, uint32_t wg_n_B,
        uint32_t sg_m, uint32_t sg_n_A, uint32_t sg_n_B, uint32_t sg_k_A,
        uint32_t sg_k_B, uint32_t wg_n_B_total, mem_layout layout_a,
        mem_layout layout_state_a, mem_layout layout_state_b,
        mem_layout layout_c, mem_space mem_space_temp, uint32_t local_kslicing,
        mma_engine engine, uint32_t periodic_sync_interval_A,
        uint32_t prefetch_distance_A, uint32_t periodic_sync_interval_B,
        uint32_t prefetch_distance_B, gpu_arch arch_tag, bool unaligned = false,
        bool temp_in_reg = false>
struct lora_gemm_fused {
    static constexpr uint32_t local_range_m = (wg_m + sg_m - 1) / sg_m;
    static constexpr uint32_t local_range_nA = (wg_n_A + sg_n_A - 1) / sg_n_A;
    static constexpr uint32_t local_range_nB = (wg_n_B + sg_n_B - 1) / sg_n_B;
    static constexpr uint32_t local_range_n
            = local_range_nA > local_range_nB ? local_range_nA : local_range_nB;
    static constexpr uint32_t num_threads
            = local_range_m * local_range_n * local_kslicing;

    using gemm_lora_a_t = gemm_lora_a<dtype_a, dtype_b, dtype_a, dtype_acc,
            wg_m, wg_n_A, sg_m, sg_n_A, sg_k_A, layout_a, layout_state_a,
            layout_a, mem_space::global, mem_space::global, mem_space_temp,
            local_kslicing, engine, periodic_sync_interval_A,
            prefetch_distance_A, arch_tag, unaligned, temp_in_reg>;

    using matTemp_t = subgroup::tile_t<dtype_a,
            typename gemm_lora_a_t::matAcc_t::tile_desc>;

    static constexpr uint32_t gemm_b_n_iters
            = (wg_n_B_total + wg_n_B - 1) / wg_n_B;

    using gemm_lora_b_t = gemm_lora_b<dtype_a, dtype_b, dtype_c, dtype_acc,
            wg_m, wg_n_B, sg_m, sg_n_B, sg_k_B, wg_n_B_total, layout_a,
            layout_state_b, layout_c, mem_space_temp, mem_space::global,
            mem_space::global, local_kslicing, engine, periodic_sync_interval_B,
            prefetch_distance_B, arch_tag, unaligned, temp_in_reg, matTemp_t>;

    static constexpr uint32_t barrier_count = gemm_lora_a_t::barrier_count + 1;
    static_assert(
            gemm_lora_b_t::barrier_count == 0, "barrier_count should be 0");
    static constexpr uint32_t slm_size = gemm_lora_a_t::slm_size;
    static_assert(gemm_lora_b_t::slm_size == 0, "slm_size should be 0");

    inline static void run(sycl::nd_item<3> &item, uint32_t m, uint32_t k,
            uint32_t n, uint32_t lora_rank, dtype_a *lora_input,
            dtype_b *state_a, dtype_b *state_alpha, dtype_b *state_b,
            dtype_c *out, dtype_a *lora_temp XETLA_POST_OP_ARGS) {

        typename gemm_lora_a_t::matAcc_t matAcc;
        gemm_lora_a_t::run(item, m, k, lora_rank, lora_input, state_a,
                lora_temp, state_alpha, matAcc);

#if LORA_TEMP_IN_REG == 1
        matTemp_t matTemp;
        subgroup::elemwise_cvt(matTemp, matAcc);
#endif

        if (LORA_TEMP_IN_REG != 1) {
            xetla_nbarrier_t<num_threads, num_threads, arch_tag> nbarrier;
            nbarrier.init_nbarrier(
                    barrier_count - 1, nbarrier_role::producer_consumer);
            xetla_fence<memory_kind::untyped_global>();
            nbarrier.arrive_wait();
        }
#pragma unroll
        for (uint32_t i = 0; i < gemm_b_n_iters; i++) {
            gemm_lora_b_t::run(item, m, lora_rank, n, lora_temp, state_b, out, i
#if LORA_TEMP_IN_REG == 1
                    ,
                    matTemp
#endif
                            XETLA_POST_OP_ARGS_PASS);
        }
    }
};
