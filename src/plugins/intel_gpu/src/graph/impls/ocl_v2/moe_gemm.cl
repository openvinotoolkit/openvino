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
#define DECORATOR moe
#include "expert_gemm_common.cl"

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global INPUT0_TYPE *input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
        const global uchar *weight_ptr,
#else
        const global INPUT1_TYPE *weight_ptr,
#endif
        global OUTPUT_TYPE *out_ptr,
#ifdef POST_PROC_SILU_MUL
        const global OUTPUT_TYPE *post_op_input,
#endif
        const global INPUT2_TYPE *experts_ids,
        const global INPUT3_TYPE *input_offset_per_expert,
        const global INPUT4_TYPE *n_array,
        int m, int k
#ifdef BIAS_DT
        , const global BIAS_DT *bias_ptr
#endif
#ifdef WEIGHT_COMPRESSED_INT4
        , const global WEIGHT_SCALE_DT *weight_scales
        #ifdef WEIGHT_ZP_DT
        , const global WEIGHT_ZP_DT *weight_zps
        #endif
#endif
#ifdef USE_SLM
        , local int* slm
#endif
) {
    // --- Preamble: per-expert-batch dispatch ---
    uint batch = get_group_id(2);
    int input_offset = sub_group_broadcast(input_offset_per_expert[batch], 0);

    #ifdef IS_GENERATE
    if (INPUT_SEQ_LEN > 1) {
    #endif
    input_ptr += input_offset * INPUT_STRIDE;
    #ifdef IS_GENERATE
    }
    #endif
    out_ptr += input_offset * OUTPUT_STRIDE;
#ifdef POST_PROC_SILU_MUL
    post_op_input += input_offset * OUTPUT_STRIDE;
#endif
    INPUT2_TYPE expert_id = sub_group_broadcast(experts_ids[batch], 0);
    int cur_n_tokens = sub_group_broadcast(n_array[batch], 0);

    // --- Shared GEMM computation (sets c_tile_half, sg_i0, sg_j0) ---
#include "expert_gemm_compute.cl"

    // --- Contiguous tile store ---
    tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
}
