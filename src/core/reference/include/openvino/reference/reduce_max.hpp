// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <numeric>

#include "openvino/reference/utils/coordinate_transform.hpp"
#include "shape_util.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of ReduceMax operator.
 *
 * @param in             Input pointer to data.
 * @param out            Output pointer to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
template <class T>
void reduce_max(const T* in, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    constexpr auto min_value = std::numeric_limits<T>::lowest();

    const auto out_shape = util::reduce(in_shape, reduction_axes);
    std::fill(out, std::next(out, shape_size(out_shape)), min_value);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const auto& in_coord : input_transform) {
        constexpr u_int64_t init_value = 0;
        const auto out_coord = util::reduce(in_coord, reduction_axes);

        const auto in_idx = std::inner_product(in_coord.begin(), in_coord.end(), in_strides.begin(), init_value);
        const auto out_idx = std::inner_product(out_coord.begin(), out_coord.end(), out_strides.begin(), init_value);

        out[out_idx] = std::max(out[out_idx], in[in_idx]);
    }
}
}  // namespace reference
}  // namespace ov
