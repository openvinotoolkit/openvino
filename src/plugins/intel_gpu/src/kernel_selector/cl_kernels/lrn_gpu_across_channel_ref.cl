// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL (lrn_gpu_across_channel_ref)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#if   defined OUTPUT_LAYOUT_BFYX
    const uint x            = get_global_id(0);
    const uint y            = get_global_id(1);
    const uint b_f          = get_global_id(2);
    const uint batch_id     = b_f / INPUT0_FEATURE_NUM;
    const uint feature_id   = b_f % INPUT0_FEATURE_NUM;

    if (x >= INPUT0_SIZE_X)
        return;
#else
    const uint b_f          = get_global_id(0);
    const uint x            = (uint)get_global_id(1);
    const uint y            = (uint)get_global_id(2);
    const uint feature_id   = b_f / INPUT0_BATCH_NUM;
    const uint batch_id     = b_f % INPUT0_BATCH_NUM;
#endif

    const uint input_id = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

    INPUT0_TYPE acc = INPUT0_VAL_ZERO;

    int input_offset_f = feature_id - PADDING;
    int input_idx = (int)input_id - PADDING*INPUT0_FEATURE_PITCH;

    for (int i = 0; i < LOCAL_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;

        INPUT0_TYPE value = zero ? INPUT0_VAL_ZERO : TO_INPUT0_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f++;
        input_idx += INPUT0_FEATURE_PITCH;
    }
    acc = mad(acc, TO_INPUT0_TYPE(ALPHA_DIV_BY_SIZE), TO_INPUT0_TYPE(K));
    acc = native_powr(acc, -TO_INPUT0_TYPE(BETA));

    const uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    INPUT0_TYPE lrn_result = acc * input[input_id];

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[output_idx] = res;
#else
    output[output_idx] = ACTIVATION(lrn_result, ACTIVATION_PARAMS);
#endif

}
