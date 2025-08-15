// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL (normalize_gpu_within_spatial_ref)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    const __global SCALE_TABLE_TYPE* scale_input
)
{
    const uint gws0 = get_global_id(0);
    const uint gws1 = get_global_id(1);
    const uint b = get_global_id(2);
    uint f, y, x = 0;

    // Compute norm
    uint input_idx;
    float norm = EPSILON;
#if NORM_AXIS == 1
    x = gws0;
    y = gws1;
    for (f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        input_idx = INPUT0_GET_INDEX(b, f , y, x);
        float value = (float)input[input_idx];
        norm = mad(value, value, norm);

    }
#elif NORM_AXIS == 2
    x = gws0;
    f = gws1;
    for (y = 0; y < INPUT0_SIZE_Y; y++)
    {
        input_idx = INPUT0_GET_INDEX(b, f, y, x);
        float value = (float)input[input_idx];
        norm = mad(value, value, norm);
    }
#elif NORM_AXIS == 3
    y = gws0;
    f = gws1;
    for (x = 0; x < INPUT0_SIZE_X; x++)
    {
        input_idx = INPUT0_GET_INDEX(b, f, y, x);
        float value = (float)input[input_idx];
        norm = mad(value, value, norm);
    }
#else
#error "Unsupported NORM_AXIS value"
#endif
    uint output_idx;

    if(norm <= THRESHOLD)
    {
        norm = 0;
    }
    else
    {
        norm = native_powr(norm, -0.5f);
    }

#if NORM_AXIS == 1
    for (f = 0; f < INPUT0_FEATURE_NUM; f++)
#elif NORM_AXIS == 2
    for (y = 0; y < INPUT0_SIZE_Y; y++)
#elif NORM_AXIS == 3
    for (x = 0; x < INPUT0_SIZE_X; x++)
#else
#error "Unsupported NORM_AXIS value"
#endif
    {
        output_idx =  OUTPUT_GET_INDEX(b, f, y, x);
        input_idx =  INPUT0_GET_INDEX(b, f, y, x);
#if SCALE_TABLE_FEATURE_NUM == 1
        const uint scale_index = 0;
#elif INPUT0_FEATURE_NUM <= SCALE_TABLE_FEATURE_NUM
        const uint scale_index = SCALE_TABLE_GET_INDEX(0, f, 0, 0);
#else
        const uint scale_index = SCALE_TABLE_GET_INDEX(0, f % SCALE_TABLE_FEATURE_NUM, 0, 0);
#endif
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(norm) * TO_ACTIVATION_TYPE(input[input_idx]) * TO_ACTIVATION_TYPE(scale_input[scale_index]);
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif

    }
}
