// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(gather_tree_gpu_ref)(
    const __global INPUT0_TYPE* step_input,
    const __global INPUT1_TYPE* parent_input,
    const __global INPUT2_TYPE* max_seq_len_input,
    const __global INPUT3_TYPE* end_token,
    __global OUTPUT_TYPE* output)
{
    const int beam = get_global_id(0);
    const int batch = get_global_id(1);
    /*
         b -> time
         f -> batch
         y -> beam
    */

    const int max_sequence_in_beam = min(INPUT0_BATCH_NUM, (int)max_seq_len_input[batch]);
    int time;
    for (time = INPUT0_BATCH_NUM - 1; time >= max_sequence_in_beam; time--) {
        output[OUTPUT_GET_INDEX(time, batch, beam, 0)] = TO_OUTPUT_TYPE(end_token[0]);
    }

    for (int parent = beam; time >= 0; time--) {
        output[OUTPUT_GET_INDEX(time, batch, beam, 0)] = TO_OUTPUT_TYPE(step_input[INPUT0_GET_INDEX(time, batch, parent, 0)]);
        parent = (int)parent_input[INPUT1_GET_INDEX(time, batch, parent, 0)];
    }
    bool finished = false;
    for (int time = 0; time < max_sequence_in_beam; time++) {
        if (finished) {
            output[OUTPUT_GET_INDEX(time, batch, beam, 0)] = TO_OUTPUT_TYPE(end_token[0]);
        } else if (output[OUTPUT_GET_INDEX(time, batch, beam, 0)] == TO_OUTPUT_TYPE(end_token[0])) {
            finished = true;
        }
    }
}
