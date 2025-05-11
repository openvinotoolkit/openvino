// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

KERNEL(deformable_convolution_gpu_bfyx_ref)(
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
#if DEFORMABLE_MODE
    , const  __global INPUT1_TYPE* trans
#if DEFORMABLE_MASK_ENABLED
    , const  __global INPUT2_TYPE* mask
#endif
#endif
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint of = get_global_id(2);
    const uint b = 0;
#else
    const uint of = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif

    UNIT_TYPE dotProd = UNIT_VAL_ZERO;
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const int channel_per_deformable_group = INPUT0_FEATURE_NUM / DEFORMABLE_GROUPS / FILTER_GROUPS_NUM;
    const uint out_index = GET_DATA_INDEX(OUTPUT, b, of, y, x);

#if GROUPED
    const uint f = of % FILTER_OFM_NUM;
    const uint conv_group = of / FILTER_OFM_NUM;
    const uint in_split_offset = conv_group * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint f = of;
    const uint conv_group = 0;
    const uint in_split_offset = conv_group * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif

    const int offset_size = FILTER_SIZE_Y * FILTER_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;

    for (uint c = 0; c < FILTER_IFM_NUM; ++c) {
#if DEFORMABLE_GROUPS == 1
        const int deformable_group_idx = 0;
#else
        const int deformable_group_idx = (FILTER_IFM_NUM * of + c) / (( FILTER_IFM_NUM * FILTER_GROUPS_NUM) / DEFORMABLE_GROUPS) % DEFORMABLE_GROUPS;
#endif
        const int trans_offset = b * INPUT1_BATCH_PITCH + deformable_group_idx * 2 * offset_size;
#if DEFORMABLE_MASK_ENABLED
        const int mask_offset = b * INPUT2_BATCH_PITCH + deformable_group_idx * offset_size;
#endif
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;

            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                const int trans_y_idx = ((2 * (j * FILTER_SIZE_X + i)) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
                float transformed_y = input_offset_y + (float)trans[trans_offset + trans_y_idx];
                const int input_offset_x = input_x + i * DILATION_SIZE_X;

                const int trans_x_idx = ((2 * (j * FILTER_SIZE_X + i) + 1) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
                float transformed_x = input_offset_x + (float)trans[trans_offset + trans_x_idx];

#if BILINEAR_INTERPOLATION_PAD
                const bool x_is_out_of_boundaries = (int)transformed_x >= INPUT0_SIZE_X || (int)transformed_x <= -1;
                const bool y_is_out_of_boundaries = (int)transformed_y >= INPUT0_SIZE_Y || (int)transformed_y <= -1;
#else
                const bool x_is_out_of_boundaries = transformed_x >= INPUT0_SIZE_X || transformed_x < 0;
                const bool y_is_out_of_boundaries = transformed_y >= INPUT0_SIZE_Y || transformed_y < 0;
#endif
#if DEFORMABLE_MASK_ENABLED
                const int mask_idx = mask_offset + ((j * FILTER_SIZE_X + i) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
#endif
                uint ifm = c;
                uint filter_idx = GET_FILTER_INDEX(FILTER, conv_group, f, ifm, j, i);

                int top_y_index    = (int)(floor(transformed_y));
                int left_x_index   = (int)(floor(transformed_x));

                if (!y_is_out_of_boundaries && !x_is_out_of_boundaries) {
#if BILINEAR_INTERPOLATION_PAD
                    int bottom_y_index = top_y_index + 1;
                    int right_x_index  = left_x_index + 1;

                    INPUT0_TYPE top_left     = top_y_index < 0 || left_x_index < 0 ? 0 :
                        (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, left_x_index) + in_split_offset];
                    INPUT0_TYPE top_right    = top_y_index < 0 || right_x_index >= INPUT0_SIZE_X ? 0 :
                        (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, right_x_index) + in_split_offset];
                    INPUT0_TYPE bottom_left  = bottom_y_index >= INPUT0_SIZE_Y || left_x_index < 0 ? 0 :
                        (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, left_x_index) + in_split_offset];
                    INPUT0_TYPE bottom_right = bottom_y_index >= INPUT0_SIZE_Y || right_x_index >= INPUT0_SIZE_X ? 0 :
                        (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, right_x_index) + in_split_offset];
#else
                    int bottom_y_index = (int)(min(ceil(transformed_y), (float)INPUT0_SIZE_Y - 1));
                    int right_x_index  = (int)(min(ceil(transformed_x), (float)INPUT0_SIZE_X - 1));

                    INPUT0_TYPE top_left     = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, left_x_index) + in_split_offset];
                    INPUT0_TYPE top_right    = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, right_x_index) + in_split_offset];
                    INPUT0_TYPE bottom_left  = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, left_x_index) + in_split_offset];
                    INPUT0_TYPE bottom_right = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, right_x_index) + in_split_offset];
#endif
                    INPUT0_TYPE top    = top_left + (top_right - top_left) * (transformed_x - left_x_index);
                    INPUT0_TYPE bottom = bottom_left + (bottom_right - bottom_left) * (transformed_x - left_x_index);

                    INPUT0_TYPE value  = top + (bottom - top) * (transformed_y - top_y_index);

#if DEFORMABLE_MASK_ENABLED
                    dotProd += value * weights[filter_idx] * (float)mask[mask_idx];
#else
                    dotProd += value * weights[filter_idx];
#endif
                } // !y_is_out_of_boundaries && !x_is_out_of_boundaries
            } // i
        } // j
    } // c

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, of, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = of;
#endif
    dotProd += (UNIT_TYPE)biases[bias_index];
#endif
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, of, y, x);
    output[dst_index] = ACTIVATION(dotProd, ACTIVATION_PARAMS);
}
