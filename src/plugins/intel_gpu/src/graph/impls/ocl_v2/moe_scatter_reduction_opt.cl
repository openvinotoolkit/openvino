// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/fetch_utils.cl"

#define VLOAD CAT(vload, VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)
#define INPUT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_BLK_SIZE)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_BLK_SIZE)

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
    const uint token_group_id = (uint)get_group_id(0);
    const uint threads_index = (uint)get_local_id(0);

     OUTPUT_VEC_TYPE output_vec[BATCHES_PER_THREAD];
    // start_offset_idx[i] = n : info for i-th expert in this thread is in the nth slot of the mask
    __local uint start_offset_index[ACTIVE_EXPERTS];
    __local uint input_offset;

    if (threads_index < ACTIVE_EXPERTS) {
        INPUT1_TYPE expert_id = experts_per_token[token_group_id * ACTIVE_EXPERTS  + threads_index];
        for (int i = 0; i < INPUT6_BATCH_NUM; i++) {
             if (experts_ids[i] == expert_id) {
                start_offset_index[threads_index] = i;
                break;
            }
        }
    }

    if (threads_index == 0)
        input_offset = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint dest_index = token_group_id * HIDDEN_SIZE;
    uint output_pos = dest_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;

    for (uint i = 0; i < BATCHES_PER_THREAD; i++) {
        output_vec[i] = TO_OUTPUT_TYPE(0);
    }

    for (uint i = 0; i < ACTIVE_EXPERTS; i++) {
        INPUT1_TYPE expert_id = experts_per_token[token_group_id * ACTIVE_EXPERTS  + i];
        INPUT2_TYPE expert_weight = expert_weights[token_group_id * ACTIVE_EXPERTS  + i];
        INPUT5_TYPE token_len = tokens_len_per_expert[start_offset_index[i]];
        INPUT4_TYPE expert_offset = experts_start_offset[start_offset_index[i]];

        for (uint tid = threads_index; tid < token_len; tid += get_local_size(0)) {
            if (tokens_per_expert[expert_offset + tid] == token_group_id) {
                input_offset = expert_offset + tid;
                break;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
            const uint input_pos = input_offset * HIDDEN_SIZE + j * VEC_BLK_SIZE + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;
                INPUT_VEC_TYPE input_data = VLOAD(0, &input[input_pos]);
                input_data *= expert_weight;
                output_vec[j] += input_data;
        }
    }

    for (uint v = 0; v < BATCHES_PER_THREAD; v++) {
        const uint out_pos = output_pos + v * VEC_BLK_SIZE;
        VSTORE(output_vec[v], 0, &output[out_pos]);
    }
}