// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL (lrn_gpu_within_channel)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    for (uint index = get_global_id(0) ; index < INPUT0_LENGTH ; index += get_global_size(0))
    {
#if   defined OUTPUT_LAYOUT_YXFB
        const uint batch_id   = index % INPUT0_BATCH_NUM;
        const uint yxf        = index / INPUT0_BATCH_NUM;
        const uint feature_id = yxf   % INPUT0_FEATURE_NUM;
        const uint yx         = yxf   / INPUT0_FEATURE_NUM;
        const uint x          = yx    % INPUT0_SIZE_X;
        const uint y          = yx    / INPUT0_SIZE_X;
#elif defined OUTPUT_LAYOUT_BFYX
        const uint x          = index % INPUT0_SIZE_X;
        const uint bfy        = index / INPUT0_SIZE_X;
        const uint y          = bfy   % INPUT0_SIZE_Y;
        const uint bf         = bfy   / INPUT0_SIZE_Y;
        const uint feature_id = bf    % INPUT0_FEATURE_NUM;
        const uint batch_id   = bf    / INPUT0_FEATURE_NUM;
#endif

        const uint first_index_in_feature = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH;
        const uint input_id = first_index_in_feature + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

        int wstart = x - PADDING;
        int hstart = y - PADDING;
        int hend = min(hstart + LOCAL_SIZE, INPUT0_SIZE_Y + PADDING);
        int wend = min(wstart + LOCAL_SIZE, INPUT0_SIZE_X + PADDING);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, (int)0);
        wstart = max(wstart, (int)0);
        hend = min(hend, INPUT0_SIZE_Y);
        wend = min(wend, INPUT0_SIZE_X);
        INPUT0_TYPE aveval = 0;

        __global const INPUT0_TYPE* bottom_slice = input + first_index_in_feature;
        for (int h = hstart; h < hend; ++h)
        {
            for (int w = wstart; w < wend; ++w)
            {
                INPUT0_TYPE tmp_val = bottom_slice[h*INPUT0_Y_PITCH + w*INPUT0_X_PITCH] * TO_INPUT0_TYPE(ALPHA_VAL_FACTOR);
                aveval += (tmp_val * tmp_val);
            }
        }

        INPUT0_TYPE acc = aveval / pool_size;
        acc = mad(acc, TO_INPUT0_TYPE(ALPHA_AFTER_FACTORED), TO_INPUT0_TYPE(K));
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
}
