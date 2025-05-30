// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/reference/reduce_sum.hpp"

namespace ov::reference {

/**
 * @brief Reference implementation of ReduceMean operator.
 *
 * @param in             Input pointer to data.
 * @param out            Output pointer to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
template <class T>
void reduce_mean(const T* in, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    reduce_sum(in, out, in_shape, reduction_axes);

    if (const auto input_size = shape_size(in_shape); input_size != 0) {
        const auto out_shape = util::reduce(in_shape, reduction_axes);
        const auto out_size = shape_size(out_shape);
        const auto count = input_size / out_size;
        std::transform(out, std::next(out, out_size), out, [count = static_cast<T>(count)](auto&& value) -> T {
            return value / count;
        });
    }
}
}  // namespace ov::reference
