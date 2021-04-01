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

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(deformable_convolution_gpu_bfyx_interp)(
    const __global INPUT0_TYPE* data,
    const __global INPUT1_TYPE* trans,
    __global OUTPUT_TYPE* output)
{
    const int xy = get_global_id(0);
    const int x = xy % OUTPUT_SIZE_X;
    const int y = xy / OUTPUT_SIZE_X;
    const int dg = (uint)get_global_id(1) % DEFORMABLE_GROUPS;
    const int b  = (uint)get_global_id(1) / DEFORMABLE_GROUPS;
    const int kw = (uint)get_global_id(2) % FILTER_SIZE_X;
    const int kh = (uint)get_global_id(2) / FILTER_SIZE_X;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const int channel_per_deformable_group = INPUT0_FEATURE_NUM / DEFORMABLE_GROUPS;

    const int trans_offset = b * INPUT1_BATCH_PITCH +
                             dg * 2 * FILTER_SIZE_Y * FILTER_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int input_offset_x = input_x + kw * DILATION_SIZE_X;
    const int input_offset_y = input_y + kh * DILATION_SIZE_Y;
    const int trans_x_idx = ((2 * (kh * FILTER_SIZE_X + kw) + 1) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
    const int trans_y_idx = ((2 * (kh * FILTER_SIZE_X + kw)) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
    float transformed_x = input_offset_x + trans[trans_offset + trans_x_idx];
    float transformed_y = input_offset_y + trans[trans_offset + trans_y_idx];
    const bool zero_y = transformed_y >= INPUT0_SIZE_Y || transformed_y < 0;
    const bool zero_x = transformed_x >= INPUT0_SIZE_X || transformed_x < 0;

    int top_y_index    = (int)(floor(transformed_y));
    int bottom_y_index = (int)(min(ceil(transformed_y), (float)INPUT0_SIZE_Y - 1));
    int left_x_index   = (int)(floor(transformed_x));
    int right_x_index  = (int)(min(ceil(transformed_x), (float)INPUT0_SIZE_X - 1));

    int oc = kh*FILTER_SIZE_X*INPUT0_FEATURE_NUM + kw*INPUT0_FEATURE_NUM + dg*channel_per_deformable_group;
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, oc, y, x);
    if (!zero_y & !zero_x & xy < OUTPUT_SIZE_X*OUTPUT_SIZE_Y) {
        for (int c = 0; c < channel_per_deformable_group; ++c) {
            uint ifm = dg * channel_per_deformable_group + c;

            INPUT0_TYPE top_left     = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, left_x_index)];
            INPUT0_TYPE top_right    = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, right_x_index)];
            INPUT0_TYPE bottom_left  = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, left_x_index)];
            INPUT0_TYPE bottom_right = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, right_x_index)];

            INPUT0_TYPE top    = top_left + (top_right - top_left) * (transformed_x - left_x_index);
            INPUT0_TYPE bottom = bottom_left + (bottom_right - bottom_left) * (transformed_x - left_x_index);

            output[dst_index + c*OUTPUT_FEATURE_PITCH] = top + (bottom - top) * (transformed_y - top_y_index);
        }
    } else if (xy < OUTPUT_SIZE_X*OUTPUT_SIZE_Y) {
        for (int c = 0; c < channel_per_deformable_group; ++c) {
            output[dst_index + c*OUTPUT_FEATURE_PITCH] = 0.0f;
        }
    }

}
