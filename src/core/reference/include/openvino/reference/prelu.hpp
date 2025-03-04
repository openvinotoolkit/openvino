// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov::reference {
namespace func {
// Usage of custom function instead of lambda, gives smaller binary size.
template <class T>
T prelu(const T x, const T y) {
    if constexpr (std::is_unsigned_v<T>) {
        return x;
    } else {
        return x < T(0) ? x * y : x;
    }
}
}  // namespace func

template <typename T>
void prelu(const T* arg, const T* slope, T* out, const Shape& arg_shape, const Shape& slope_shape) {
    Shape slope_shape_tmp = slope_shape;
    const auto channel_dim_idx = arg_shape.size() > 1 ? 1 : 0;
    if (slope_shape.size() == 1 && arg_shape[channel_dim_idx] == slope_shape[0]) {
        Shape channel_slope_shape(arg_shape.size(), 1);
        channel_slope_shape[channel_dim_idx] = slope_shape[0];
        std::swap(slope_shape_tmp, channel_slope_shape);
    }
    autobroadcast_binop(arg, slope, out, arg_shape, slope_shape_tmp, op::AutoBroadcastType::NUMPY, func::prelu<T>);
}
}  // namespace ov::reference
