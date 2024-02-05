// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(beam_table_update)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* state_prev,
    __global const INPUT1_TYPE* beam_idx,
    __global OUTPUT_TYPE* state_new,
    uchar is_state_set)
{
    const unsigned int b = (uint)get_global_id(0);
    const unsigned int s = (uint)get_global_id(1);

    const unsigned int out_offset = b * OUTPUT_BATCH_PITCH + s;
    const unsigned int in_offset = beam_idx[b] * INPUT0_BATCH_PITCH + s;

    if (s >= OUTPUT_BATCH_PITCH)
        return;

    if (!is_state_set) {
        state_new[out_offset] = TO_OUTPUT_TYPE(b);
    } else {
        if (s < INPUT0_BATCH_PITCH) {
            state_new[out_offset] = state_prev[in_offset];
        } else {
            state_new[out_offset] = b;
        }
    }
}
