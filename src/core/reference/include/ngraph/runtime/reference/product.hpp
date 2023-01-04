// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void product(const T* arg, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    constexpr bool dont_keep_dims_in_output = false;
    const auto out_shape = reduce(in_shape, reduction_axes, dont_keep_dims_in_output);
    std::fill(out, out + shape_size(out_shape), T(1));

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const Coordinate& input_coord : input_transform) {
        const Coordinate output_coord = reduce(input_coord, reduction_axes, dont_keep_dims_in_output);

        const size_t in_idx =
            std::inner_product(input_coord.begin(), input_coord.end(), in_strides.begin(), uint64_t(0));
        const size_t out_idx =
            std::inner_product(output_coord.begin(), output_coord.end(), out_strides.begin(), uint64_t(0));

        out[out_idx] = out[out_idx] * arg[in_idx];
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
