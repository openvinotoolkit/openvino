// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(reorder_kv_cache)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* state,
    __global const INPUT1_TYPE* src_idx,
    __global const INPUT2_TYPE* dst_idx,
    __global OUTPUT_TYPE* state_new,,
    uint seq_len)
{
    const unsigned int bh = (uint)get_global_id(0);
    const unsigned int d = (uint)get_global_id(1);

    for (int s = 0; s < IXD_LEN; ++s) {
        const unsigned int out_offset = bh * OUTPUT_SEQ_PITCH * seq_len + dst_idx[s * OUTPUT_SEQ_PITCH] * OUTPUT_SEQ_PITCH + d;
        const unsigned int in_offset = bh * INPUT0_SEQ_PITCH * seq_len + src_idx[s] * INPUT0_SEQ_PITCH + d;

        state_new[out_offset] = state[in_offset];
    }
}
