// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(col2im_opt)(const __global INPUT0_TYPE* input,
                                 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    const uint batch = get_global_id(2);
    const uint channel_idx = get_global_id(0);

    const int channel_offset = batch * NUM_CHANNELS + channel_idx;

    for (int idx = 0; idx < KERNEL_PRODUCT; ++idx) {
        const int width_offset = idx % KERNEL_SIZE_Y;
        const int height_offset = (idx / KERNEL_SIZE_Y) % KERNEL_SIZE_X;
        const int column = channel_idx * KERNEL_PRODUCT + idx;
        const int column_offset = batch * NUM_ELEMENTS_FOR_BLOCK + column;

        for (int column_height_idx = 0; column_height_idx < ORIG_HEIGHT; ++column_height_idx) {
            const int image_height_idx = column_height_idx * STRIDE_SIZE_X - PAD_BEGIN_SIZE_X + height_offset * DILATION_SIZE_X;

            if (image_height_idx >= 0 && image_height_idx < OUT_SIZE_X) {
                for (int column_width_idx = 0; column_width_idx < ORIG_WIDTH; ++column_width_idx) {
                    const int image_width_idx = column_width_idx * STRIDE_SIZE_Y - PAD_BEGIN_SIZE_Y + width_offset * DILATION_SIZE_Y;

                    if (image_width_idx >= 0 && image_width_idx < OUT_SIZE_Y) {
                        const int img_idx = (channel_offset * OUT_SIZE_X + image_height_idx) * OUT_SIZE_Y + image_width_idx;
                        const int data_idx = (column_offset * ORIG_HEIGHT + column_height_idx) * ORIG_WIDTH + column_width_idx;

                        // sum the overlapping values
                        output[img_idx] += input[data_idx];
                    }
                }
            }
        }
    }
}
