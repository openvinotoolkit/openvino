// Copyright (C) 2018-2026 Intel Corporation
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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint b = get_global_id(2);

    // Compute norm
    uint input_idx;
    float norm = EPSILON;
    for (int i = 0; i < INPUT0_FEATURE_NUM; i++)
    {
        input_idx = INPUT0_GET_INDEX(b, i , y, x);
        float value = (float)input[input_idx];
        norm = mad(value, value, norm);

    }

    uint output_idx;

    norm = rsqrt(norm);

    for (int f = 0; f < INPUT0_FEATURE_NUM; f++)
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
        // Normalize in f32: the scale of a slice with a tiny norm (1/sqrt(eps) = 1e6 for an all-zero slice) can exceed
        // the f16 range, the normalized value can't.
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(norm * (float)input[input_idx] * (float)scale_input[scale_index]);
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif

    }
}
