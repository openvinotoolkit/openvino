// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

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
             const char* off_value) {
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
            auto input_val = indices[outer_i + inner_i];
            // Negative indices are ignored
            if ((input_val >= 0) && (static_cast<size_t>(input_val) < depth)) {
                auto oh_index = static_cast<size_t>(input_val);
                size_t output_offset = out_elem_size * (outer_i * depth + inner_i + oh_index * inner_block);
                std::copy(on_value, on_value + out_elem_size, out + output_offset);
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
