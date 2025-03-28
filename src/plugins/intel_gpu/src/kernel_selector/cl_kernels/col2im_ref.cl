// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(col2im_ref)(const __global INPUT0_TYPE* input,
                                 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    const uint output_size[2] = {OUT_SIZE_X, OUT_SIZE_Y};
    const uint kernel_size[2] = {KERNEL_SIZE_X, KERNEL_SIZE_Y};
    const uint strides[2] = {STRIDE_SIZE_X, STRIDE_SIZE_Y};
    const uint dilations[2] = {DILATION_SIZE_X, DILATION_SIZE_Y};
    const uint pads_begin[2] = {PAD_BEGIN_SIZE_X, PAD_BEGIN_SIZE_Y};

    const uint num_blocks = INPUT0_SIZE_Y;
    const uint kernel_product = KERNEL_SIZE_X * KERNEL_SIZE_Y;
    const uint channels_per_column = INPUT0_FEATURE_NUM;
    const uint channel_count = channels_per_column / KERNEL_PRODUCT;

    const uint batch = get_global_id(2);

    for (int column = 0; column < channels_per_column; ++column) {
        const int width_offset = column % kernel_size[1];
        const int height_offset = (column / kernel_size[1]) % kernel_size[0];
        const int channel_idx = column / kernel_product;

        for (int column_height_idx = 0; column_height_idx < ORIG_HEIGHT; ++column_height_idx) {
            const int image_height_idx = column_height_idx * strides[0] - pads_begin[0] + height_offset * dilations[0];
            if (image_height_idx >= 0 && image_height_idx < output_size[0]) {
                for (int column_width_idx = 0; column_width_idx < ORIG_WIDTH; ++column_width_idx) {
                    const int image_width_idx = column_width_idx * strides[1] - pads_begin[1] + width_offset * dilations[1];
                    if (image_width_idx >= 0 && image_width_idx < output_size[1]) {
                        const int img_idx = ((batch * channel_count + channel_idx) * output_size[0] + image_height_idx) * output_size[1] + image_width_idx;
                        const int data_idx = ((batch * channels_per_column + column) * ORIG_HEIGHT + column_height_idx) * ORIG_WIDTH + column_width_idx;

                        // sum the overlapping values
                        output[img_idx] += input[data_idx];
                    }
                }
            }
        }
    }
}
