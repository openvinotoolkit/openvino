// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
/// \brief Return number of non-zero entries in the input argument.
///
/// \param arg Input tensor
/// \param arg_shape Input tensor shape
/// Output number of non-zero entries in arg
template <typename T>
size_t non_zero_get_count(const T* arg, const Shape& arg_shape) {
    const auto zero = T{0};
    const auto arg_count = shape_size(arg_shape);
    return arg_count - std::count(arg, arg + arg_count, zero);
}

/// \brief Return indices of non-zero entries in input argument.
///
/// \param arg Input tensor
/// \param arg_shape Input tensor shape
/// \param out Output containing indices of non-zero entries in arg
template <typename T, typename U>
void non_zero(const T* arg, U* out, const Shape& arg_shape) {
    const auto non_zero_count = non_zero_get_count(arg, arg_shape);
    if (non_zero_count == 0)
        return;

    const auto arg_count = shape_size(arg_shape);
    if (arg_count == 1) {
        out[0] = U{0};
        return;
    }

    // Dimensional size for the arg_shape. This is used to map one-dimentional
    // arg array indices to corresponding arg_rank-dimentional shape indices.
    // i.e., arg_shape {2, 3, 2}
    // Array index 4 in arg (arg[4]) correspond to 3-D index of [0][2][0]
    const auto arg_strides = row_major_strides(arg_shape);
    const auto arg_transform = CoordinateTransformBasic{arg_shape};

    // Find non-zero entries in arg and write the indices info in out.
    // For a non-zero entry, map its array index to corresponding indices
    // in arg_shape, then, write each of the arg_shape index value to
    // out array at the position that is determined by entry's dimension
    // distance and number of non-zero entries prior to it.
    // i,e., Given input with shape{2, 3, 2}, rank = 3
    // input [[[0, 2], [0, 0], [3, 4]],
    //        [[5, 0], [6, 0], [7, 8]]]
    //
    // output shape {3, 7}, rank = 2
    // output [[0, 0, 0, 1, 1, 1, 1],
    //         [0, 2, 2, 0, 1, 2, 2],
    //         [1, 0, 1, 0, 0, 0, 1]]
    //
    // input[0][2][0] = 3 is arg[4]
    // output for this entry out[1] = 0, out[8] = 2, out[15] = 0
    const auto arg_rank = arg_shape.size();
    size_t col_index = 0;
    for (const auto& arg_coord : arg_transform) {
        const auto arg_index = coordinate_offset(arg_coord, arg_strides);
        if (arg[arg_index] != T{0}) {
            for (size_t j = 0, out_index = col_index; j < arg_rank; ++j, out_index += non_zero_count) {
                out[out_index] = static_cast<U>(arg_coord[j]);
            }
            ++col_index;
        }
    }
}
}  // namespace reference
}  // namespace ov
