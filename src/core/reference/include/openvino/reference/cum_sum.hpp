// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

namespace ov {
namespace reference {

template <typename T, typename P>
void cumsum(const T* arg, const P axis, T* out, const Shape& tensor_shape, const bool exclusive, const bool reverse) {
    const auto rank = tensor_shape.size();
    const auto normalized_axis = axis >= 0 ? axis : rank + axis;
    const auto axis_dim = tensor_shape[normalized_axis];

    const auto size_before_axis = shape_size(tensor_shape.begin(), tensor_shape.begin() + normalized_axis);
    const auto size_after_axis = shape_size(tensor_shape.begin() + normalized_axis + 1, tensor_shape.end());

    const auto reverse_shift = reverse ? -1 : 1;
    const auto element_shift = exclusive ? size_after_axis * reverse_shift : 0;

    for (size_t i = 0; i < size_before_axis; ++i) {
        const auto slice_idx = i * axis_dim * size_after_axis + reverse * size_after_axis * (axis_dim - 1);
        for (size_t j = 0; j < size_after_axis; ++j) {
            const auto sequence_start_idx = slice_idx + j;
            out[sequence_start_idx] = exclusive ? T{0} : arg[sequence_start_idx];
            for (size_t k = 1; k < axis_dim; ++k) {
                const auto element_idx = sequence_start_idx + (k * size_after_axis) * reverse_shift;
                const auto in_idx = element_idx - element_shift;
                const auto previous_sum_idx = element_idx - size_after_axis * reverse_shift;
                out[element_idx] = out[previous_sum_idx] + arg[in_idx];
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
