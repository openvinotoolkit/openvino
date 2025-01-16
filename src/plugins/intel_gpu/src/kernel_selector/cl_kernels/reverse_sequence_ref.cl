// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(reverse_sequence_ref)(const __global INPUT0_TYPE* input, const __global INPUT1_TYPE* seq_lengths, __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    const uint y = (uint)get_global_id(2) / INPUT0_SIZE_X;
    const uint x = (uint)get_global_id(2) % INPUT0_SIZE_X;
    uint dimensions[] = { batch, feature, y, x };

    const uint input_index = INPUT0_GET_INDEX(batch, feature, y, x);

    const uint length = (uint)seq_lengths[dimensions[BATCH_AXIS]];
    if (dimensions[SEQ_AXIS] < length)
        dimensions[SEQ_AXIS] = length - dimensions[SEQ_AXIS] - 1;

    const uint output_index = OUTPUT_GET_INDEX(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}
