// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(feed_forward_gpu_ref)(
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
    const __global INPUT2_TYPE* input2,
    const __global INPUT3_TYPE* input3,
    const __global INPUT4_TYPE* input4,
    __global OUTPUT_TYPE* output)
{

#if OUTPUT_DIMS == 5
    uint data_idx = (uint)get_global_id(GWS_YX);
    const uint x = data_idx % OUTPUT_SIZE_X;
    data_idx = data_idx / OUTPUT_SIZE_X;
    const uint y = data_idx % OUTPUT_SIZE_Y;
    data_idx = data_idx / OUTPUT_SIZE_Y;
    const uint z = data_idx % OUTPUT_SIZE_Z;
#else // 2D spatial
    const uint x = (uint)get_global_id(GWS_YX) % OUTPUT_SIZE_X;
    const uint y = (uint)get_global_id(GWS_YX) / OUTPUT_SIZE_X;
#endif
    const uint f = (uint)get_global_id(GWS_FEATURE);
    const uint b = (uint)get_global_id(GWS_BATCH);

#if OUTPUT_DIMS == 5
    const uint output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
    const uint input0_idx = INPUT0_GET_INDEX(b, f, z, y, x);
#else // 2D spatial
    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
    const uint input0_idx = INPUT0_GET_INDEX(b, f, y, x);
#endif

#if IS_SCALAR
    const uint input_idx = 0;
#else
    const uint input_idx = input0_idx;
#endif

    ACCUMULATOR_TYPE res = TO_ACCUMULATOR_TYPE(input0[input0_idx]);
    res = (tanh(input2[input_idx] * (input1[input_idx] * res * res * res + res)) + input3[input_idx]) * res * input4[input_idx];
    output[output_idx] = TO_OUTPUT_TYPE(res);
}
