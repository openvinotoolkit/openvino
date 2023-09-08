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
template <typename T>
void reduce_min(const T* in, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    constexpr auto max_value =
        std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();

    const auto out_shape = util::reduce(in_shape, reduction_axes);
    std::fill(out, out + shape_size(out_shape), max_value);

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const auto& in_coord : input_transform) {
        constexpr uint64_t init_value = 0;
        const auto out_coord = util::reduce(in_coord, reduction_axes);

        const auto in_idx = std::inner_product(in_coord.begin(), in_coord.end(), in_strides.begin(), init_value);
        const auto out_idx = std::inner_product(out_coord.begin(), out_coord.end(), out_strides.begin(), init_value);

        out[out_idx] = std::min(out[out_idx], in[in_idx]);
    }
}
}  // namespace reference
}  // namespace ov
