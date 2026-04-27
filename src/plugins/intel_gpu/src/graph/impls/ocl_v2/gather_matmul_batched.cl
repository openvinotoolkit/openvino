/*******************************************************************************
 * Copyright 2025 Intel Corporation
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
 */

#include "include/batch_headers/common.cl"
#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"
#define DECORATOR gm
#include "expert_gemm_common.cl"
#include "expert_gemm_compute.cl"

// Batched GatherMatmul kernel — grouped GEMM with scattered output.
//
// Instead of dispatching one workgroup per (token, expert_slot) pair (GEMV),
// this kernel dispatches workgroups per (group, token_tile, m_tile) where
// each group contains all tokens routed to the same expert in the same slot.
// Weight matrix is loaded once and shared across all tokens in the group.
//
// Input: gathered_A[total_sorted_tokens, K] — contiguous per-group (from bgm_gather)
// Weights: B[n_all_experts, N, K] — expert weights (transposed)
// Output: out[top_k, n_tokens, N] — scattered writes via token_map
//
// Dispatch:
//   x = ceil_div(M, wg_tile_m)           — output feature tiles
//   y = ceil_div(n_tokens, wg_tile_n)    — token tiles (early-exit per group)
//   z = max_groups                        — group dimension (early-exit)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(gather_matmul_batched)(
    OPTIONAL_SHAPE_INFO_ARG
    const global half* gathered_input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
    const global uchar* weight_ptr,
#else
    const global INPUT1_TYPE* weight_ptr,
#endif
    global OUTPUT_TYPE* out_ptr,
    const global int* group_expert_ids,
    const global int* group_slot_ids,
    const global int* group_offsets,
    const global int* group_sizes,
    const global int* token_map,
    const global int* num_groups,
    int m,
    int k
#ifdef BIAS_DT
    ,
    const global BIAS_DT* bias_ptr
#endif
#ifdef WEIGHT_COMPRESSED_INT4
    ,
    const global WEIGHT_SCALE_DT* weight_scales
#    ifdef WEIGHT_ZP_DT
    ,
    const global WEIGHT_ZP_DT* weight_zps
#    endif
#endif
#ifdef USE_SLM
    ,
    local int* slm
#endif
) {
    // --- Preamble: decode group, compute pointers ---
    uint group_id = get_group_id(2);
    if (group_id >= (uint)num_groups[0])
        return;  // early-exit for empty group slots

    int expert_id = group_expert_ids[group_id];
    int slot = group_slot_ids[group_id];
    int offset = group_offsets[group_id];
    int cur_n_tokens = group_sizes[group_id];
    int n_tokens = N_TOKENS;

    const global half* input_ptr = gathered_input_ptr + offset * k;

    // --- Shared GEMM computation ---
    UGEMM_C_TYPE_HALF c_tile_half;
    uint sg_i0, sg_j0;
    if (!expert_gemm_compute(input_ptr, weight_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
                             weight_scales,
#    ifdef WEIGHT_ZP_DT
                             weight_zps,
#    endif
#endif
#ifdef BIAS_DT
                             bias_ptr,
#endif
#ifdef USE_SLM
                             slm,
#endif
                             expert_id, cur_n_tokens, m, k,
                             &c_tile_half, &sg_i0, &sg_j0))
        return;

    // --- Scattered store: write each row to original token position ---
    {
        int sglid = get_sub_group_local_id();
        const int br = UGEMM_C_TYPE_BLOCK0;
        const int nbr = UGEMM_C_TYPE_NBLOCK0;
        const int bc = UGEMM_C_TYPE_BLOCK1;
        const int nbc = UGEMM_C_TYPE_NBLOCK1;
        int sg = SUBGROUP_SIZE;

        unroll_for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < cur_n_tokens) {
                int orig_token = token_map[offset + sg_j0 + j];
                // out[slot, orig_token, :] — row pointer
                global OUTPUT_TYPE* row_ptr = out_ptr + slot * n_tokens * m + orig_token * m;
                unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    int i = i0 + sglid;
                    if (sg_i0 + i < m) {
                        row_ptr[sg_i0 + i] = c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br) / sg + (j % bc) * (br / sg)];
                    }
                }
            }
        }
    }
}
