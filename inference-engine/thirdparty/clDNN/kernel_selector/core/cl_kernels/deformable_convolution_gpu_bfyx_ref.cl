// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/include_all.cl"

KERNEL(deformable_convolution_gpu_bfyx_ref)(
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if DEFORMABLE_MODE
    const  __global INPUT1_TYPE* trans,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif

    UNIT_TYPE dotProd = UNIT_VAL_ZERO;
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const int channel_per_deformable_group = INPUT0_FEATURE_NUM / DEFORMABLE_GROUPS;
    const uint out_index = GET_DATA_INDEX(OUTPUT, b, f, y, x);

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif

    for (uint dg = 0; dg < DEFORMABLE_GROUPS; ++dg)
    {
        const int ifm_offset = dg * channel_per_deformable_group;
        const int trans_offset = b * INPUT1_BATCH_PITCH +
                                 dg * 2 * FILTER_SIZE_Y * FILTER_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                const int trans_y_idx = ((2 * (j * FILTER_SIZE_X + i)) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
                float transformed_y = input_offset_y + (float)trans[trans_offset + trans_y_idx];

                const bool zero_y = transformed_y >= INPUT0_SIZE_Y || transformed_y < 0;

                const int input_offset_x = input_x + i * DILATION_SIZE_X;

                const int trans_x_idx = ((2 * (j * FILTER_SIZE_X + i) + 1) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
                float transformed_x = input_offset_x + (float)trans[trans_offset + trans_x_idx];

                const bool zero_x = transformed_x >= INPUT0_SIZE_X || transformed_x < 0;
                if(!zero_y) {
                    if(!zero_x) {
                        for (uint c = 0; c < channel_per_deformable_group; ++c) {
                            uint ifm = ifm_offset + c;
                            uint filter_idx = GET_FILTER_INDEX(FILTER, f, ifm, j, i);

#if GROUPED && !DEPTHWISE_SEPARABLE_OPT
    filter_idx += split_idx * FILTER_LENGTH;
#endif
#ifdef LOCAL_CONVOLUTION
    filter_idx += FILTER_SIZE_X * FILTER_SIZE_Y * (x + OUTPUT_SIZE_X * y);
#endif

                            int top_y_index    = (int)(floor(transformed_y));
                            int bottom_y_index = (int)(min(ceil(transformed_y), (float)INPUT0_SIZE_Y - 1));
                            int left_x_index   = (int)(floor(transformed_x));
                            int right_x_index  = (int)(min(ceil(transformed_x), (float)INPUT0_SIZE_X - 1));

                            INPUT0_TYPE top_left     = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, left_x_index) + in_split_offset];
                            INPUT0_TYPE top_right    = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, right_x_index) + in_split_offset];
                            INPUT0_TYPE bottom_left  = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, left_x_index) + in_split_offset];
                            INPUT0_TYPE bottom_right = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, right_x_index) + in_split_offset];

                            INPUT0_TYPE top    = top_left + (top_right - top_left) * (transformed_x - left_x_index);
                            INPUT0_TYPE bottom = bottom_left + (bottom_right - bottom_left) * (transformed_x - left_x_index);

                            INPUT0_TYPE value  = top + (bottom - top) * (transformed_y - top_y_index);

                            dotProd += value * weights[filter_idx];
                        }
                    } // zero_x
                } // zero_y
            } // i

        } // j
    } // dg

#if BIAS_TERM
#if GROUPED && !DEPTHWISE_SEPARABLE_OPT
        const uint bias_offset = split_idx * BIAS_LENGTH;
#else
        const uint bias_offset = 0;
#endif
#if   BIAS_PER_OUTPUT
        const uint bias_index = bias_offset + GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
        const uint bias_index = bias_offset + f;
#endif
        dotProd += (UNIT_TYPE)biases[bias_index];
#endif
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y, x) + out_split_offset;
    output[dst_index] = ACTIVATION(dotProd, ACTIVATION_PARAMS);
}
