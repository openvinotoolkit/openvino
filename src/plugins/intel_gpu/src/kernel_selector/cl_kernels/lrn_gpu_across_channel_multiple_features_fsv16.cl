// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

REQD_SUB_GROUP_SIZE(16)
KERNEL (lrn_gpu_across_channel_multiple_features_fsv16)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint feature_id   = (uint)get_global_id(0);
    const uint x            = (uint)get_global_id(1);
    const uint b_y          = (uint)get_global_id(2);
    const uint batch_id     = b_y / INPUT0_SIZE_Y;
    const uint y            = b_y % INPUT0_SIZE_Y;

    if (feature_id >= INPUT0_FEATURE_NUM)
        return;

    int input_offset_f = feature_id - PADDING;

    INPUT0_TYPE val[LOCAL_SIZE];
    INPUT0_TYPE res = 0;
    for (uint i = 0; i < LOCAL_SIZE; ++i, ++input_offset_f) {
        bool non_zero = input_offset_f >= 0 && input_offset_f < INPUT0_FEATURE_NUM;
        uint input_idx = INPUT0_GET_INDEX(batch_id, max(input_offset_f, (int)0), y, x);
        val[i] = (int)non_zero * TO_INPUT0_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * TO_INPUT0_TYPE(input[input_idx]);
        res = mad(val[i], val[i], res);
    }
    res = mad(res, TO_INPUT0_TYPE(ALPHA_DIV_BY_SIZE), TO_INPUT0_TYPE(K));
    res = native_powr(res, -TO_INPUT0_TYPE(BETA));

    uint output_idx = OUTPUT_GET_INDEX(batch_id, feature_id, y, x);
    uint input_idx = INPUT0_GET_INDEX(batch_id, feature_id, y, x);
    INPUT0_TYPE lrn_result = res * input[input_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
    #else
        output[output_idx] = ACTIVATION(TO_OUTPUT_TYPE(lrn_result), ACTIVATION_PARAMS);
    #endif
}
