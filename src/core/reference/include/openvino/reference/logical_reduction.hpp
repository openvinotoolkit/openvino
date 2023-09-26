// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "ngraph/shape_util.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
OPENVINO_SUPPRESS_DEPRECATED_START
static inline void reduce_logical_and(const char* arg,
                                      char* out,
                                      const Shape& in_shape,
                                      const AxisSet& reduction_axes) {
    constexpr bool dont_keep_dims_in_output = false;
    const auto out_shape = ngraph::reduce(in_shape, reduction_axes, dont_keep_dims_in_output);
    std::fill(out, out + shape_size(out_shape), 1);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const Coordinate& input_coord : input_transform) {
        const Coordinate output_coord = ngraph::reduce(input_coord, reduction_axes, dont_keep_dims_in_output);

        const size_t in_idx =
            std::inner_product(input_coord.begin(), input_coord.end(), in_strides.begin(), uint64_t(0));
        const size_t out_idx =
            std::inner_product(output_coord.begin(), output_coord.end(), out_strides.begin(), uint64_t(0));

        out[out_idx] = out[out_idx] && arg[in_idx];
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END

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
