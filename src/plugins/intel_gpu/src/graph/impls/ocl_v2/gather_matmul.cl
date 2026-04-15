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
    // --- Preamble: decode (token_idx, expert_slot), compute pointers ---
    uint flat_idx = get_group_id(2);
    uint top_k = TOP_K;
    uint token_idx = flat_idx / top_k;
    uint expert_slot = flat_idx % top_k;
    int n_tokens = N_TOKENS;

    int expert_id = sub_group_broadcast(indices[token_idx * top_k + expert_slot], 0);

    uint n_act = N_ACTIVATED_EXPERTS;
    uint a_slot = min(expert_slot, n_act - 1);
    input_ptr += a_slot * n_tokens * INPUT_STRIDE + token_idx * INPUT_STRIDE;
    out_ptr += expert_slot * n_tokens * OUTPUT_STRIDE + token_idx * OUTPUT_STRIDE;

    // n=1 per dispatch — each workgroup processes a single token. Used for the
    // decode path; prefill uses gather_matmul_batched.cl, which groups tokens by
    // expert so each workgroup amortizes one weight load across many tokens.
    int cur_n_tokens = 1;

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

    // --- Contiguous tile store ---
    tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
}
