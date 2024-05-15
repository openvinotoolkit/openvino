// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T, typename T_idx>
void col2im(const T* data,
            const Shape& data_shape,
            const T_idx* output_size,
            const T_idx* kernel_size,
            T* out,
            const Strides& strides,
            const Strides& dilations,
            const Shape& pads_begin,
            const Shape& pads_end) {
    // calculate the original height and width
    const auto original_height =
        (output_size[0] + pads_begin[0] + pads_end[0] - (dilations[0] * (kernel_size[0] - 1) + 1)) / strides[0] + 1;
    const auto original_width =
        (output_size[1] + pads_begin[1] + pads_end[1] - (dilations[1] * (kernel_size[1] - 1) + 1)) / strides[1] + 1;

    // fill output with zeros to account for missing dilated values
    const auto kernel_product = kernel_size[0] * kernel_size[1];
    const bool is_batched = data_shape.size() == 3;
    const int64_t C_idx = is_batched ? 1 : 0;
    const auto channel_count = data_shape[C_idx] / kernel_product;
    const auto batch_count = is_batched ? data_shape[0] : 1;
    std::fill_n(out, batch_count * output_size[0] * output_size[1] * channel_count, T(0));

    const auto channels_per_column = channel_count * kernel_product;
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        for (int64_t column = 0; column < channels_per_column; ++column) {
            const auto width_offset = column % (int64_t)kernel_size[1];
            const auto height_offset = (column / (int64_t)kernel_size[1]) % (int64_t)kernel_size[0];
            const auto channel_idx = column / (int64_t)kernel_size[0] / (int64_t)kernel_size[1];

            for (int64_t column_height_idx = 0; column_height_idx < original_height; ++column_height_idx) {
                const int64_t image_height_idx =
                    column_height_idx * strides[0] - pads_begin[0] + height_offset * dilations[0];

                for (int64_t column_width_idx = 0; column_width_idx < original_width; ++column_width_idx) {
                    const int64_t image_width_idx =
                        column_width_idx * strides[1] - pads_begin[1] + width_offset * dilations[1];

                    if (image_height_idx >= 0 && image_height_idx < output_size[0] && image_width_idx >= 0 &&
                        image_width_idx < output_size[1]) {
                        int64_t img_idx = ((batch * channel_count + channel_idx) * output_size[0] + image_height_idx) *
                                              output_size[1] +
                                          image_width_idx;
                        int64_t data_idx =
                            ((batch * channels_per_column + column) * original_height + column_height_idx) *
                                original_width +
                            column_width_idx;

                        // sum the overlapping values
                        out[img_idx] += data[data_idx];
                    }
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
