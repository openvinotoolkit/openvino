// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(slice_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
#if INPUT0_DIMS <= 4
    const uint xy = get_global_id(2);
    const uint y = xy / OUTPUT_SIZE_X;
    const uint x = xy % OUTPUT_SIZE_X;
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, y, x);
    const uint input_index = INPUT0_GET_INDEX(
        SLICE_BEGIN_BATCH + batch * SLICE_STEP_BATCH,
        SLICE_BEGIN_FEATURE + feature * SLICE_STEP_FEATURE,
        SLICE_BEGIN_Y + y * SLICE_STEP_Y,
        SLICE_BEGIN_X + x * SLICE_STEP_X);
#elif INPUT0_DIMS == 5
    const uint xyz = get_global_id(2);
    const uint yx = xyz % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = xyz / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
    const uint input_index = INPUT0_GET_INDEX(
        SLICE_BEGIN_BATCH + batch * SLICE_STEP_BATCH,
        SLICE_BEGIN_FEATURE + feature * SLICE_STEP_FEATURE,
        SLICE_BEGIN_Z + z * SLICE_STEP_Z,
        SLICE_BEGIN_Y + y * SLICE_STEP_Y,
        SLICE_BEGIN_X + x * SLICE_STEP_X);
#endif
    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}
