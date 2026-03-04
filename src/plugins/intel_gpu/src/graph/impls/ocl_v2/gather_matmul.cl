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

// BatchGatherMatmul kernel — per-token dispatch
//
// A: [n_activated_experts, n_tokens, hidden_size] — input activations
//    n_activated_experts is 1 for the first GatherMatmul (broadcast), or top_k for subsequent ones
// B: [n_all_experts, N, K] — expert weights (transposed)
// indices: [n_tokens, top_k] — per-token expert indices for weight gathering
// Output: [top_k, n_tokens, N]
//
// Dispatch:
//   get_group_id(0) = M tile (output features)
//   get_group_id(1) = 0 (single token per dispatch, y-dim unused)
//   get_group_id(2) = flat index over (token_idx, expert_slot) pairs
//                     token_idx = get_group_id(2) / TOP_K
//                     expert_slot = get_group_id(2) % TOP_K
//
// For each (token_idx, expert_slot):
//   expert_id = indices[token_idx * top_k + expert_slot]
//   a_slot = min(expert_slot, n_activated_experts - 1)  (clamp for broadcast case)
//   output[expert_slot, token_idx, :] = A[a_slot, token_idx, :] @ B[expert_id, :, :]^T

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(batch_gather_matmul)(OPTIONAL_SHAPE_INFO_ARG const global INPUT0_TYPE* input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
                                                                                      const global uchar* weight_ptr,
#else
                                                                                      const global INPUT1_TYPE* weight_ptr,
#endif
                                                                                      global OUTPUT_TYPE* out_ptr,
                                                                                      const global INPUT2_TYPE* indices,
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
    // Decode (token_idx, expert_slot) from flat z-dim index
    uint flat_idx = get_group_id(2);
    uint top_k = TOP_K;
    uint token_idx = flat_idx / top_k;
    uint expert_slot = flat_idx % top_k;
    int n_tokens = N_TOKENS;

    // Per-token expert lookup: indices[token_idx * top_k + expert_slot]
    INPUT2_TYPE expert_id = sub_group_broadcast(indices[token_idx * top_k + expert_slot], 0);

    // Input A indexing: A[a_slot, token_idx, :]
    // a_slot = min(expert_slot, n_activated_experts - 1) to handle broadcast case (A dim[0] = 1)
    uint n_act = N_ACTIVATED_EXPERTS;
    uint a_slot = min(expert_slot, n_act - 1);
    input_ptr += a_slot * n_tokens * INPUT_STRIDE + token_idx * INPUT_STRIDE;

    // Output indexing: out[expert_slot, token_idx, :]
    out_ptr += expert_slot * n_tokens * OUTPUT_STRIDE + token_idx * OUTPUT_STRIDE;

    // Weight indexing: B[expert_id, :, :]
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

    // @todo n=1 per dispatch — each workgroup processes a single token.
    // This is correct but not optimal for prefill. Future optimization: sort tokens
    // by expert within each slot and batch tokens that share the same expert.
    int cur_n_tokens = 1;

    if (wg_j0 >= cur_n_tokens)
        return; /* early exit if outside token range */
#ifdef IS_GENERATE
#    ifdef USE_SLM
    ugemm_gm_c_type c_tile = ugemm_gm(weight_ptr,
                                        ld_weight,
                                        input_ptr,
                                        ld_input,
                                        m,
                                        cur_n_tokens,
                                        k,
                                        wg_i0,
                                        wg_j0,
                                        0,
                                        sg_i,
                                        sg_j,
                                        sg_k,
                                        slm
#    else
    ugemm_gm_c_type c_tile = ugemm_gm(weight_ptr,
                                        ld_weight,
                                        input_ptr,
                                        ld_input,
                                        m,
                                        cur_n_tokens,
                                        k,
                                        wg_i0,
                                        wg_j0,
                                        0,
                                        sg_i,
                                        sg_j,
                                        sg_k,
                                        0
#    endif
#else
#    ifdef USE_SLM
    ugemm_gm_c_type c_tile = ugemm_gm(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, slm
#    else
    ugemm_gm_c_type c_tile = ugemm_gm(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, 0
#    endif
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
    bias_ptr += (expert_id * BIAS_STRIDE);
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
                    c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br) / sg + (j % bc) * (br / sg)] += bias_ptr[sg_i0 + i];
                }
            }
        }
    }
#endif

    // No POST_PROC_SILU_MUL — activation functions are separate ops in the GatherMatmul graph.

    tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
}
