// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
/**
 * @brief Reference implementation of ReduceMin operator.
 *
 * @param in             Input pointer to data.
 * @param out            Output pointer to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
template <class T>
void reduce_min(const T* in, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    constexpr auto max_value =
        std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();

    const auto out_shape = util::reduce(in_shape, reduction_axes);
    std::fill(out, out + shape_size(out_shape), max_value);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const auto& in_coord : input_transform) {
        const auto out_coord = util::reduce(in_coord, reduction_axes);
        const auto in_idx = coordinate_offset(in_coord, in_strides);
        const auto out_idx = coordinate_offset(out_coord, out_strides);

        out[out_idx] = std::min(out[out_idx], in[in_idx]);
    }
}
}  // namespace reference
}  // namespace ov
