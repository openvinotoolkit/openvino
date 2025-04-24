// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"

REQD_SUB_GROUP_SIZE(16)
__attribute__((reqd_work_group_size(16, 1, 1)))
KERNEL(convolution_depthwise_weights_lwg)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
)
{
    const uint yx = (uint)get_global_id(0);
    const uint x = yx % OUTPUT_SIZE_X;
    const uint y = yx / OUTPUT_SIZE_X;
    const uint f = (uint)get_global_id(1);
    const uint b = (uint)get_global_id(2);

    UNIT_TYPE dotProd = UNIT_VAL_ZERO;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint g = (f / FILTER_OFM_NUM);
    const uint in_group_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    const uint filter_offset = f*FILTER_GROUPS_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_group_offset;

#if FILTER_SIZE_Y * FILTER_SIZE_X % 16 == 0 && !FP16_UNIT_USED
    UNIT_TYPE w = DT_FILTER_BLOCK_READ(weights, filter_offset);
#elif FILTER_SIZE_X * FILTER_SIZE_Y > 16 && FILTER_SIZE_X * FILTER_SIZE_Y <= 25
    const uint lid = get_local_id(0);
    UNIT_TYPE w[2] = { UNIT_VAL_ZERO };
    w[0] = weights[filter_offset + lid];
    if (16 + lid < FILTER_SIZE_X * FILTER_SIZE_Y)
        w[1] = weights[filter_offset + 16 + lid];
#else
    const uint lid = get_local_id(0);
    UNIT_TYPE w = UNIT_VAL_ZERO;
    if (lid < FILTER_SIZE_X * FILTER_SIZE_Y)
        w = weights[filter_offset + lid];
#endif

    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
    for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
    {
        const int input_offset_y = input_y + j * DILATION_SIZE_Y;
#if BOUNDARY_CHECK
        const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

        if(!zero_y)
        {
#endif
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                const int input_offset_x = input_x + i * DILATION_SIZE_X;
#if BOUNDARY_CHECK
                const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                if(!zero_x)
                {
#endif
#if FILTER_SIZE_X * FILTER_SIZE_Y > 16 && FILTER_SIZE_X * FILTER_SIZE_Y <= 25
                    const uint id = (j*FILTER_Y_PITCH + i*FILTER_X_PITCH) / 16;
                    const uint idx = (j*FILTER_Y_PITCH + i*FILTER_X_PITCH) % 16;
                    UNIT_TYPE w1 = _sub_group_shuffle(w[id], idx);
#else
                    UNIT_TYPE w1 = _sub_group_shuffle(w, j*FILTER_Y_PITCH + i*FILTER_X_PITCH);
#endif
                    dotProd = mad(input[input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH],
                                  w1, dotProd);
                }
            }
#if BOUNDARY_CHECK
        }
    }
#endif

    if(yx >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
        return;

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif
    dotProd += (UNIT_TYPE)biases[bias_index];
#endif

    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y, x);
    output[dst_index] = ACTIVATION(dotProd, ACTIVATION_PARAMS);

}
