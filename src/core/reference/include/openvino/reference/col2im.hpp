// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"
#include "openvino/op/roi_align.hpp"
//#include "openvino/reference/utils/coordinate_index.hpp"
//#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {

template <typename T>
void col2im(const T* data,
            const Shape& data_shape,
            const int64_t* output_size,
            const int64_t* kernel_size,
            T* out,
            const Strides& strides,
            const Strides& dilations,
            const Shape& pads_begin,
            const Shape& pads_end) {

    const bool is_batched = data_shape.size() == 3;
    const size_t C_idx = is_batched ? 1 : 0;
    const auto elements_per_block = data_shape[C_idx];
    const auto kernel_product = (*kernel_size[0]) * (*kernel_size[1]);
    const auto channel_count = elements_per_block / kernel_product;
    const auto image_height = *output_size[0];
    const auto image_width = *output_size[1];
    const auto column_size = (*output_size[0]) * (*output_size[1]);  // ???
    const auto channels_per_column = channel_count * kernel_product;

    // fill output with zeros to account for missing dilated values
    std::fill_n(out, image_height * image_width * channel_count, T(0));

    for (size_t column = 0; column < channels_per_column; ++column) {
        const auto height_offset = column % kernel_size[0];
        const auto width_offset = (column / kernel_size[1]) % kernel_size[0];
        const auto channel_idx = column / kernel_size[0] / kernel_size[1];

        for (size_t column_height_idx = 0; column_height_idx < image_height; ++column_height_idx) {
            const size_t image_height_idx = column_height_idx * strides[0] - pads_begin[0] - pads_end[0] + height_offset * dilations[0];

            for (size_t column_width_idx = 0; column_width_idx < image_width; ++column_width_idx) {
                const size_t image_width_idx = column_width_idx * strides[1] - pads_begin[1] - pads_end[1] + width_offset * dilations[1];

                if (image_height_idx >= 0 && image_height_idx < image_height && image_width_idx >= 0 && image_width_idx < image_width) {
                    const auto img_idx = image_width * (channel_idx * image_height * image_height_idx) + image_width_idx;
                    const auto data_idx = image_width * (column * image_height + image_height_idx) + column_width_idx;
                    // addition is necessary because overlapping values need to be summed
                    out[img_idx] += data[data_idx]
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
