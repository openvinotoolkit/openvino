// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(col2im)(const __global INPUT0_TYPE* input,
                    __global OUTPUT_TYPE* output
)
{
    const uint batch = get_global_id(2);
    const uint channel_idx = get_global_id(0);

    const int channel_offset = (batch * NUM_CHANNELS + channel_idx) * OUT_SIZE_0;

    for (int idx = 0; idx < KERNEL_PRODUCT; ++idx) {
        const int width_offset = (idx % KERNEL_SIZE_1) * DILATION_1 - PAD_BEGIN_1;
        const int height_offset = ((idx / KERNEL_SIZE_1) % KERNEL_SIZE_0) * DILATION_0 - PAD_BEGIN_0;
        const int max_width = MIN((ORIG_WIDTH * STRIDE_1), (OUT_SIZE_1 - width_offset));
        const int max_height = MIN((ORIG_HEIGHT * STRIDE_0), (OUT_SIZE_0-height_offset));
        const int min_width = MAX(((-width_offset)/STRIDE_1)*STRIDE_1, 0);
        const int min_height = MAX(((-height_offset)/STRIDE_0)*STRIDE_0, 0);

        for (int image_height_idx = min_height; image_height_idx < max_height; image_height_idx+=STRIDE_0) {
            for (int image_width_idx = min_width; image_width_idx < max_width; image_width_idx+=STRIDE_1) {
                const int img_idx = (channel_offset + image_height_idx + height_offset) * OUT_SIZE_1 + image_width_idx + width_offset;
                output[img_idx] = 0;
            }
        }
    }

    for (int idx = 0; idx < KERNEL_PRODUCT; ++idx) {
        const int width_offset = (idx % KERNEL_SIZE_1) * DILATION_1 - PAD_BEGIN_1;
        const int height_offset = ((idx / KERNEL_SIZE_1) % KERNEL_SIZE_0) * DILATION_0 - PAD_BEGIN_0;
        const int column = channel_idx * KERNEL_PRODUCT + idx;
        const int column_offset = batch * NUM_ELEMENTS_FOR_BLOCK + column;

        for (int column_height_idx = 0; column_height_idx < ORIG_HEIGHT; ++column_height_idx) {
            const int image_height_idx = column_height_idx * STRIDE_0 + height_offset;

            if (image_height_idx >= 0 && image_height_idx < OUT_SIZE_0) {
                for (int column_width_idx = 0; column_width_idx < ORIG_WIDTH; ++column_width_idx) {
                    const int image_width_idx = column_width_idx * STRIDE_1 + width_offset;

                    if (image_width_idx >= 0 && image_width_idx < OUT_SIZE_1) {
                        const int img_idx = (channel_offset + image_height_idx) * OUT_SIZE_1 + image_width_idx;
                        const int data_idx = (column_offset * ORIG_HEIGHT + column_height_idx) * ORIG_WIDTH + column_width_idx;

                        // sum the overlapping values
                        output[img_idx] += input[data_idx];
                    }
                }
            }
        }
    }
}