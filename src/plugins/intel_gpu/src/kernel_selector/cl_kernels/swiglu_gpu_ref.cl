// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(swiglu_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
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
    const uint gate_idx = INPUT0_GET_INDEX(b, f, z, y, x);
    const uint input_idx = INPUT0_GET_INDEX(b, f, z, y, x) + SPLIT_LENGTH;
#else // 2D spatial
    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
    const uint gate_idx = INPUT0_GET_INDEX(b, f, y, x);
    const uint input_idx = INPUT0_GET_INDEX(b, f, y, x) + SPLIT_LENGTH;
#endif

    ACCUMULATOR_TYPE res = ACCUMULATOR_VAL_ZERO;

    res = (ACCUMULATOR_TYPE)input[gate_idx];
    res /= ACCUMULATOR_VAL_ONE + exp(-(ACCUMULATOR_VAL_ONE * res));
    res *= (ACCUMULATOR_TYPE)input[input_idx];

    output[output_idx] = TO_OUTPUT_TYPE(res);
}
