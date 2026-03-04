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

#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"
#ifdef BIAS_DT
DECLARE_2D_TILE(bias_tile_type, BIAS_DT, SUBGROUP_SIZE, ugemm_gm_sg_tile_m, 1, 1, 1)
#endif
DECLARE_2D_TILE(ugemm_gm_c_type_half,
                half,
                SUBGROUP_SIZE,
                ugemm_gm_c_type_block0,
                ugemm_gm_c_type_block1,
                ugemm_gm_c_type_nblock0,
                ugemm_gm_c_type_nblock1)
DECLARE_2D_TILE_COPY_REBLOCK(ugemm_gm_c_type,
                             SUBGROUP_SIZE,
                             ugemm_gm_c_type_block0,
                             ugemm_gm_c_type_block1,
                             ugemm_gm_c_type_nblock0,
                             ugemm_gm_c_type_nblock1,
                             ugemm_gm_c_type_half,
                             SUBGROUP_SIZE,
                             ugemm_gm_c_type_block0,
                             ugemm_gm_c_type_block1,
                             ugemm_gm_c_type_nblock0,
                             ugemm_gm_c_type_nblock1)

#define unroll_for __attribute__((opencl_unroll_hint)) for

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
    uint group_id = get_group_id(2);
    if (group_id >= (uint)num_groups[0])
        return;  // early-exit for empty group slots

    int expert_id = group_expert_ids[group_id];
    int slot = group_slot_ids[group_id];
    int offset = group_offsets[group_id];
    int cur_n_tokens = group_sizes[group_id];
    int n_tokens = N_TOKENS;

    // Point to gathered activations for this group
    const global half* input_ptr = gathered_input_ptr + offset * k;

    // Point to expert weights
    weight_ptr += expert_id * EXPERT_STRIDE;

    int ld_input = k;
#ifdef WEIGHT_COMPRESSED_INT4
    weight_scales += expert_id * m * NUM_GROUPS;
#    ifdef WEIGHT_ZP_DT
#        ifdef WEIGHT_COMPRESSED_ZP_INT4
    weight_zps += expert_id * m * NUM_GROUPS / 2;
#        else
    weight_zps += expert_id * m * NUM_GROUPS;
#        endif
#    endif
#endif
    int ld_weight = k;

    uint sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);
    uint sg_k = sub_group_broadcast(get_local_id(2), 0);

    uint wg_i0 = get_group_id(0) * ugemm_gm_wg_tile_m;
    uint wg_j0 = get_group_id(1) * ugemm_gm_wg_tile_n;
    uint sg_i0 = wg_i0 + sg_i * ugemm_gm_sg_tile_m;
    uint sg_j0 = wg_j0 + sg_j * ugemm_gm_sg_tile_n;
#ifdef WEIGHT_COMPRESSED_INT4
#    ifdef SCALE_ZP_NO_TRANSPOSE
    uint scale_zp_leading_dim = m;
#    else
    uint scale_zp_leading_dim = NUM_GROUPS;
#    endif
#endif

    if (wg_j0 >= cur_n_tokens)
        return;  // early exit if outside this group's token range

    // Standard ugemm micro-kernel call — identical to per-token path
    // but now cur_n_tokens > 1 (batched tokens sharing same expert weights)
#ifdef USE_SLM
    ugemm_gm_c_type c_tile = ugemm_gm(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, slm
#else
    ugemm_gm_c_type c_tile = ugemm_gm(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, 0
#endif
#ifdef WEIGHT_COMPRESSED_INT4
                                         ,
                                         weight_scales
#    ifdef WEIGHT_ZP_DT
                                         ,
                                         weight_zps
#    endif
                                         ,
                                         scale_zp_leading_dim
#endif
    );

    // Only the first sg stores data in kparallel microkernels
    if (sg_k > 0)
        return;

    ugemm_gm_c_type_half c_tile_half;
    tile_copy_reblock(c_tile, &c_tile_half);

#ifdef BIAS_DT
    const global BIAS_DT* bias_base = bias_ptr + expert_id * BIAS_STRIDE;
    int sglid = get_sub_group_local_id();
    const int br = ugemm_gm_c_type_block0;
    const int nbr = ugemm_gm_c_type_nblock0;
    const int bc = ugemm_gm_c_type_block1;
    const int nbc = ugemm_gm_c_type_nblock1;
    int sg = SUBGROUP_SIZE;
    for (int j = 0; j < bc * nbc; j++) {
        if (sg_j0 + j < cur_n_tokens) {
            for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                int i = i0 + sglid;
                if (sg_i0 + i < m) {
                    c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br) / sg + (j % bc) * (br / sg)] += bias_base[sg_i0 + i];
                }
            }
        }
    }
#endif

    // SCATTERED STORE: write each row to the original token position in output
    // out[slot, orig_token, :] instead of contiguous tile_store
    {
        int sglid = get_sub_group_local_id();
        const int br = ugemm_gm_c_type_block0;
        const int nbr = ugemm_gm_c_type_nblock0;
        const int bc = ugemm_gm_c_type_block1;
        const int nbc = ugemm_gm_c_type_nblock1;
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
