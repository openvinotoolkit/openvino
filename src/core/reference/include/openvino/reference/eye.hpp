// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Eye operator
 *
 * @param data            Pointer to output data.
 * @param out_shape       Output data size.
 * @param diagonal_index  Eye diagonal index to populate matrix with ones
 */
template <typename T>
void eye(T* data, const Shape& out_shape, const int64_t diagonal_index) {
    const auto spatial_dims_offset = out_shape.size() - 2;
    const int64_t num_columns = out_shape.back();
    const int64_t num_rows = out_shape[spatial_dims_offset];
    const int64_t matrix_size = num_rows * num_columns;
    const int64_t out_size = shape_size(out_shape);

    // fill tensor by zero
    std::fill(data, std::next(data, out_size), T(0));

    // set ones on diagonal
    constexpr int64_t zero{0};
    const auto abs_diag_idx = static_cast<int64_t>(std::abs(diagonal_index));
    const int64_t shift_by_columns = std::max(diagonal_index, zero);
    const int64_t count_by_columns = std::max(num_columns - abs_diag_idx, zero);
    const int64_t count_by_rows = std::max(num_rows - abs_diag_idx, zero);
    const int64_t count =
        diagonal_index > 0 ? std::min(count_by_columns, num_rows) : std::min(count_by_rows, num_columns);

    for (auto matrix_offset = zero; matrix_offset < out_size; matrix_offset += matrix_size) {
        for (auto j = 0; j < count; ++j) {
            const int64_t index = (j + shift_by_columns - diagonal_index) * num_columns + j + shift_by_columns;
            data[matrix_offset + index] = T{1};
        }
    }
}
}  // namespace reference
}  // namespace ov
