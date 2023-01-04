// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

#ifdef _WIN32
#    undef min
#endif

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void min(const T* arg, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    T minval =
        std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();

    constexpr bool dont_keep_dims_in_output = false;
    const auto out_shape = reduce(in_shape, reduction_axes, dont_keep_dims_in_output);
    std::fill(out, out + shape_size(out_shape), minval);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const Coordinate& input_coord : input_transform) {
        const Coordinate output_coord = reduce(input_coord, reduction_axes, dont_keep_dims_in_output);

        const size_t in_idx =
            std::inner_product(input_coord.begin(), input_coord.end(), in_strides.begin(), uint64_t(0));
        const size_t out_idx =
            std::inner_product(output_coord.begin(), output_coord.end(), out_strides.begin(), uint64_t(0));

        const T x = arg[in_idx];
        const T min = out[out_idx];
        if (x < min) {
            out[out_idx] = x;
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
