// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(convolution_gpu_yxfb_ref)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter
#if BIAS_TERM
    , const __global UNIT_TYPE* bias
#endif
)
{
    UNIT_TYPE result = UNIT_VAL_ZERO;

    const uint batch_offset = (uint)get_global_id(0) % INPUT0_BATCH_NUM;
    const uint ofm_offset   = (uint)get_global_id(0) / INPUT0_BATCH_NUM;
    const uint out_x        = (uint)get_global_id(1);
    const uint out_y        = (uint)get_global_id(2);

    const int x = (int)out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int y = (int)out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

#if GROUPED
    const uint g = ofm_offset / FILTER_OFM_NUM;
    const uint in_split_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    const uint of = ofm_offset % FILTER_OFM_NUM;
#else
    const uint g = 0;
    const uint in_split_offset = 0;
    const uint of = ofm_offset;
#endif
    const uint input_offset = INPUT0_OFFSET + batch_offset*INPUT0_BATCH_PITCH + in_split_offset;
#if GROUPED
    const uint filter_offset = (ofm_offset / FILTER_OFM_NUM) * FILTER_GROUPS_PITCH;
#else
    const uint filter_offset = 0;
#endif

    for (uint i = 0; i < FILTER_SIZE_Y; i++)
    {
        const int input_offset_y = y + i * DILATION_SIZE_Y;
        const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

        if(!zero_y)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                const int input_offset_x = x + j * DILATION_SIZE_X;
                const bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                if(!zero)
                {
                    uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH;
                    uint filter_idx = filter_offset + of*FILTER_OFM_PITCH + i*FILTER_Y_PITCH + j*FILTER_X_PITCH;

                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
                    {
                        result = fma(input[input_idx], filter[filter_idx], result);
                        filter_idx += FILTER_IFM_PITCH;
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
                }
            }
        }
    }
#if BIAS_TERM
    result += bias[ofm_offset];
#endif
    const uint out_split_offset = g * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    const uint dst_index = batch_offset*OUTPUT_BATCH_PITCH + of*OUTPUT_FEATURE_PITCH + out_y*OUTPUT_Y_PITCH + out_x*OUTPUT_X_PITCH + OUTPUT_OFFSET + out_split_offset;
    output[dst_index] = ACTIVATION(result, ACTIVATION_PARAMS);
}
