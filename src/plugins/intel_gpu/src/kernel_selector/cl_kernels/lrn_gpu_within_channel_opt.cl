// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL (lrn_gpu_within_channel_opt)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    uint index = get_global_id(0);
#if   defined OUTPUT_LAYOUT_YXFB
    const uint yxf        = index / INPUT0_BATCH_NUM;
    const uint batch_id   = index - yxf * INPUT0_BATCH_NUM;
    const uint yx         = yxf / INPUT0_FEATURE_NUM;
    const uint feature_id = yxf - yx * INPUT0_FEATURE_NUM;
    const uint y          = yx / INPUT0_SIZE_X;
    const uint x          = yx - y * INPUT0_SIZE_X;

#elif defined OUTPUT_LAYOUT_BFYX
    const uint bfy        = index / INPUT0_SIZE_X;
    const uint x          = index - bfy * INPUT0_SIZE_X;
    const uint bf         = bfy / INPUT0_SIZE_Y;
    const uint y          = bfy - bf * INPUT0_SIZE_Y;
    const uint batch_id   = bf / INPUT0_FEATURE_NUM;
    const uint feature_id = bf - batch_id * INPUT0_FEATURE_NUM;
#endif

    const uint first_index_in_feature = INPUT0_OFFSET + batch_id * INPUT0_BATCH_PITCH + feature_id * INPUT0_FEATURE_PITCH;
    const uint input_id = first_index_in_feature + y * INPUT0_Y_PITCH + x * INPUT0_X_PITCH;

    INPUT0_TYPE aveval = 0;
    uint pool_size = 0;
    int wstart = x - PADDING;
    int hstart = y - PADDING;


    if (((hstart + LOCAL_SIZE) < INPUT0_SIZE_Y) &&
        ((wstart + LOCAL_SIZE) < INPUT0_SIZE_X) &&
        (x > PADDING) &&
        (y > PADDING))
    {
        pool_size = LOCAL_SIZE * LOCAL_SIZE;

        __global const INPUT0_TYPE* bottom_slice = input + first_index_in_feature + hstart * INPUT0_Y_PITCH + wstart * INPUT0_X_PITCH;
        for (int h = 0; h < LOCAL_SIZE; ++h)
        {
            uint hPitch = h * INPUT0_Y_PITCH;
            for (int w = 0; w < LOCAL_SIZE; ++w)
            {
                INPUT0_TYPE tmp_val = bottom_slice[hPitch + w * INPUT0_X_PITCH] * TO_INPUT0_TYPE(ALPHA_VAL_FACTOR);
                aveval = mad(tmp_val, tmp_val, aveval);
            }
        }
    }
    else
    {
        int hend = min(hstart + LOCAL_SIZE, INPUT0_SIZE_Y + PADDING);
        int wend = min(wstart + LOCAL_SIZE, INPUT0_SIZE_X + PADDING);
        pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, (int)0);
        wstart = max(wstart, (int)0);
        hend = min(hend, INPUT0_SIZE_Y);
        wend = min(wend, INPUT0_SIZE_X);

        __global const INPUT0_TYPE* bottom_slice = input + first_index_in_feature;
        for (uint h = hstart; h < hend; ++h)
        {
            uint hPitch = h * INPUT0_Y_PITCH;
            for (uint w = wstart; w < wend; ++w)
            {
                INPUT0_TYPE tmp_val = bottom_slice[hPitch + w * INPUT0_X_PITCH] * TO_INPUT0_TYPE(ALPHA_VAL_FACTOR);
                aveval = mad(tmp_val, tmp_val, aveval);
            }
        }
    }

    INPUT0_TYPE acc = aveval / pool_size;
    acc = mad(acc, TO_INPUT0_TYPE(ALPHA_AFTER_FACTORED), TO_INPUT0_TYPE(K));
    acc = native_powr(acc, -TO_INPUT0_TYPE(BETA));

    const uint output_idx = OUTPUT_OFFSET + batch_id * OUTPUT_BATCH_PITCH + feature_id * OUTPUT_FEATURE_PITCH + y * OUTPUT_Y_PITCH + x * OUTPUT_X_PITCH;
    INPUT0_TYPE lrn_result = acc * input[input_id];

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[output_idx] = res;
#else
    output[output_idx] = ACTIVATION(lrn_result, ACTIVATION_PARAMS);
#endif
}
