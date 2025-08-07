// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/op/one_hot.hpp"

namespace ov {
namespace reference {
template <typename INPUT_TYPE>
void one_hot(const INPUT_TYPE* indices,
             const Shape& indices_shape,
             char* out,
             const size_t out_elem_size,
             const size_t depth,
             const int64_t one_hot_axis,
             const char* on_value,
             const char* off_value,
             const op::v1::OneHot::NegativeIndicesMode mode) {
    const bool is_mode_normalize = mode == op::v1::OneHot::NegativeIndicesMode::NORMALIZE;
    std::cout << "is_mode_normalize=" << is_mode_normalize << std::endl;
    const size_t num_ind = shape_size(indices_shape);
    // Step 1: Set off_value to the output.
    for (auto p = out; p < out + num_ind * depth * out_elem_size; p += out_elem_size)
        std::copy(off_value, off_value + out_elem_size, p);
    // Number of elements between one-hot values in the output memory layout
    const size_t inner_block = [&] {
        size_t mul = 1;
        for (size_t i = one_hot_axis; i < indices_shape.size(); ++i)
            mul *= indices_shape[i];
        return mul;
    }();
    // Step 2: Write on_value at needed positions
    for (size_t outer_i = 0; outer_i < num_ind; outer_i += inner_block) {
        for (size_t inner_i = 0; inner_i < inner_block; inner_i++) {
            const int64_t input_val = static_cast<int64_t>(indices[outer_i + inner_i]);
            const int64_t depth_i64 = static_cast<int64_t>(depth);
            const int64_t actual_index = (input_val < 0 && is_mode_normalize) ? depth_i64 + input_val : input_val;
            const int64_t max_valid = depth_i64 - 1;

            if (actual_index >= 0 && actual_index <= max_valid) {
                const size_t output_offset =
                    out_elem_size * (outer_i * depth + inner_i + static_cast<size_t>(actual_index) * inner_block);
                std::copy(on_value, on_value + out_elem_size, out + output_offset);
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
