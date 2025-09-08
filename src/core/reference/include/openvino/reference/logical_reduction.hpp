// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of ReduceLogicalAnd operator.
 *
 * @param in             Input pointer to data.
 * @param out            Output pointer to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
static inline void reduce_logical_and(const char* arg,
                                      char* out,
                                      const Shape& in_shape,
                                      const AxisSet& reduction_axes) {
    const auto out_shape = ov::util::reduce(in_shape, reduction_axes);
    std::fill(out, out + shape_size(out_shape), 1);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const auto& in_coord : input_transform) {
        const auto out_coord = ov::util::reduce(in_coord, reduction_axes);

        const auto in_idx = ov::coordinate_offset(in_coord, in_strides);
        const auto out_idx = ov::coordinate_offset(out_coord, out_strides);

        out[out_idx] = out[out_idx] && arg[in_idx];
    }
}

/**
 * @brief Reference implementation of ReduceLogicalOr operator.
 *
 * @param in             Input pointer to data.
 * @param out            Output pointer to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
static inline void reduce_logical_or(const char* arg, char* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    const auto out_shape = ov::util::reduce(in_shape, reduction_axes);
    std::fill(out, out + shape_size(out_shape), 0);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const auto& in_coord : input_transform) {
        const auto out_coord = ov::util::reduce(in_coord, reduction_axes);

        const auto in_idx = ov::coordinate_offset(in_coord, in_strides);
        const auto out_idx = ov::coordinate_offset(out_coord, out_strides);

        out[out_idx] = out[out_idx] || arg[in_idx];
    }
}
}  // namespace reference
}  // namespace ov
