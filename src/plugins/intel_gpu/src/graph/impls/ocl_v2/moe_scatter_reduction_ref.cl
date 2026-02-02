// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/fetch_utils.cl"

KERNEL(moe_scatter_reduction_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* experts_per_token,
    const __global INPUT2_TYPE* expert_weights,
    const __global INPUT3_TYPE* tokens_per_expert,
    const __global INPUT4_TYPE* experts_start_offset,
    const __global INPUT5_TYPE* tokens_len_per_expert,
    const __global INPUT6_TYPE* experts_ids,
    __global OUTPUT_TYPE* output
)
{
    uint token_id = (uint)get_global_id(0);
    uint output_base_idx = token_id * HIDDEN_SIZE;
    for (int e_iter = 0; e_iter < ACTIVE_EXPERTS; ++e_iter) {
        INPUT1_TYPE expert_id = experts_per_token[token_id * ACTIVE_EXPERTS + e_iter];
        INPUT2_TYPE weight = expert_weights[token_id * ACTIVE_EXPERTS + e_iter];
        int idx = 0;
        for (int i = 0; i < INPUT6_BATCH_NUM; ++i) {
            if (experts_ids[i] == expert_id) {
                idx = i;
                break;
            }
        }
        uint exp_offset_start = experts_start_offset[idx];
        uint input_len = tokens_len_per_expert[idx];
        uint input_offset = 0;
        for (uint t = 0; t < input_len; ++t) {
            if (tokens_per_expert[exp_offset_start + t] == token_id) {
                input_offset = exp_offset_start + t;
                break;
            }
        }
        uint in_pos = input_offset * HIDDEN_SIZE;
        uint out_pos = token_id * HIDDEN_SIZE;
        for (uint h = 0; h < HIDDEN_SIZE; h++) {
            if (e_iter == 0)
                output[out_pos + h] = input[in_pos + h] * weight;
            else
                output[out_pos + h] += input[in_pos + h] * weight;
        }
    }
}
