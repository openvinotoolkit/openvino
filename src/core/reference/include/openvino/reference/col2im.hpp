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
    // fill output with zeros to account for missing dilated values
    const auto kernel_product = kernel_size[0] * kernel_size[1];
    const bool is_batched = data_shape.size() == 3;
    const T_idx C_idx = is_batched ? 1 : 0;
    const T channel_count = data_shape[C_idx] / kernel_product;
    const T batch_count = is_batched ? data_shape[0] : 1;
    std::fill_n(out, batch_count * output_size[0] * output_size[1] * channel_count, T(0));

    // calculate the original height and width
    auto get_original_dimension = [&](const int64_t idx) {
        return (output_size[idx] + pads_begin[idx] + pads_end[idx] - (dilations[idx] * (kernel_size[idx] - 1) + 1)) /
                   strides[idx] +
               1;
    };
    const T original_height = get_original_dimension(0);
    const T original_width = get_original_dimension(1);

    auto get_image_dimension_index = [&](const T column_dim_idx, const T dim_offset, const int64_t idx) {
        return column_dim_idx * strides[idx] - pads_begin[idx] + dim_offset * dilations[idx];
    };
    const T_idx channels_per_column = channel_count * kernel_product;
    for (T batch = 0; batch < batch_count; ++batch) {
        for (T_idx column = 0; column < channels_per_column; ++column) {
            const auto width_offset = column % kernel_size[1];
            const auto height_offset = (column / kernel_size[1]) % kernel_size[0];
            const auto channel_idx = column / kernel_product;

            for (T column_height_idx = 0; column_height_idx < original_height; ++column_height_idx) {
                const T_idx image_height_idx = get_image_dimension_index(column_height_idx, height_offset, 0);

                for (T column_width_idx = 0; column_width_idx < original_width; ++column_width_idx) {
                    const T_idx image_width_idx = get_image_dimension_index(column_width_idx, width_offset, 1);

                    if (image_height_idx >= 0 && image_height_idx < output_size[0] && image_width_idx >= 0 &&
                        image_width_idx < output_size[1]) {
                        const T_idx img_idx =
                            ((batch * channel_count + channel_idx) * output_size[0] + image_height_idx) *
                                output_size[1] +
                            image_width_idx;
                        const T_idx data_idx =
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
