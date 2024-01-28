// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

KERNEL(rope_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos_sin,
    __global OUTPUT_TYPE* output)
{
    const uint p = get_global_id(0);
    const uint b = get_global_id(1);
    const uint h = get_global_id(2) / HALF_ROTARY_NDIMS;
    const uint r = get_global_id(2) % HALF_ROTARY_NDIMS * 2;

    uint input_idx = INPUT0_GET_INDEX(p, b, h * HEAD_SIZE, 0);
    uint cos_sin_idx = INPUT1_GET_INDEX(p, b, 0, 0);
    uint output_idx = OUTPUT_GET_INDEX(p, b, h, 0);

    INPUT1_TYPE cosv = cos_sin[cos_sin_idx + r];
    INPUT1_TYPE sinv = cos_sin[cos_sin_idx + r + 1];

    output[output_idx + r] = cosv * input[input_idx + r] - sinv * input[input_idx + r + 1];
    output[output_idx + r + 1] = sinv * input[input_idx + r] + cosv * input[input_idx + r + 1];

    for (uint i = HALF_ROTARY_NDIMS * 2; i < HEAD_SIZE; ++i) {
        output[output_idx + i] = input[input_idx + i];
    }
}
