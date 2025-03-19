// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(col_to_im_opt)(const __global INPUT0_TYPE* input,
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
    const uint pads_end[2] = {PAD_END_SIZE_X, PAD_END_SIZE_Y};

    const uint num_blocks = INPUT0_SIZE_Y;
    const uint kernel_product = KERNEL_SIZE_X * KERNEL_SIZE_Y;
    const uint channels_per_column = INPUT0_FEATURE_NUM; 
    const uint channel_count = channels_per_column / kernel_product;

    const uint batch_count = INPUT0_BATCH_NUM;
    const uint batch = get_global_id(2);

    // printf("batch(%d) num_blocks(%u) output(%u, %u), channel(%u, %u) original_height(%u) original_width(%u) \n",
    //         batch, num_blocks, (uint)OUT_SIZE_X, (uint)OUT_SIZE_Y, (uint)KERNEL_SIZE_X, (uint)KERNEL_SIZE_Y, ORIG_HEIGHT, ORIG_WIDTH);

    // for (uint batch = 0; batch < batch_count; ++batch) {
        for (uint column = 0; column < channels_per_column; ++column) {
            const uint width_offset = column % kernel_size[1];
            const uint height_offset = (column / kernel_size[1]) % kernel_size[0];
            const uint channel_idx = column / kernel_product;

            const uint out_idx = (batch * channel_count + channel_idx) * output_size[0];
            const uint height_idx = (batch * channels_per_column + column) * ORIG_HEIGHT;

            for (uint column_height_idx = 0; column_height_idx < ORIG_HEIGHT; ++column_height_idx) {
                // get_image_dimension_index(column_height_idx, height_offset, 0);
                const uint image_height_idx = column_height_idx * strides[0] - pads_begin[0] + height_offset * dilations[0];
                if (image_height_idx >= 0 && image_height_idx < output_size[0]) {
                    for (uint column_width_idx = 0; column_width_idx < ORIG_WIDTH; ++column_width_idx) {
                        // get_image_dimension_index(column_width_idx, width_offset, 1);
                        const uint image_width_idx = column_width_idx * strides[1] - pads_begin[1] + width_offset * dilations[1];
                        if (image_width_idx >= 0 && image_width_idx < output_size[1]) {
                            const uint img_idx = (out_idx + image_height_idx) * image_width_idx;
                            const uint data_idx = (height_idx + column_height_idx) * ORIG_WIDTH + column_width_idx;

                            // sum the overlapping values
                            output[img_idx] += input[data_idx];
                        }
                    }
                }
            }
        }
    // }
}