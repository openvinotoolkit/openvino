// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_ref.cl: Unsupported output dimension"
#endif

#define STATIC_SCALE 16.0h
KERNEL(dynamic_quantize_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    /* static_quantize version */
    // XXX: need to handle two outputs
    const uint x = (uint)get_global_id(GWS_YX) % OUTPUT_SIZE_X;
    const uint y = (uint)get_global_id(GWS_YX) / OUTPUT_SIZE_X;
    const uint f = (uint)get_global_id(GWS_FEATURE);
    const uint b = (uint)get_global_id(GWS_BATCH);

    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
    const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);

    output[output_idx] = TO_OUTPUT_TYPE(input[input_idx] * STATIC_SCALE);
}
