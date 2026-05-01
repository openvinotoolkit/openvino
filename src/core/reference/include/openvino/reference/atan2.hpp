// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T atan2(const T y_val, const T x_val) {
    return static_cast<T>(std::atan2(static_cast<double>(y_val), static_cast<double>(x_val)));
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T atan2(const T y_val, const T x_val) {
    return static_cast<T>(std::round(std::atan2(static_cast<double>(y_val), static_cast<double>(x_val))));
}

}  // namespace func

/**
 * @brief Reference implementation of binary Atan2 operator.
 *
 * @param y              Pointer to y (ordinate) input data.
 * @param x              Pointer to x (abscissa) input data.
 * @param out            Pointer to output data.
 * @param y_shape        Shape of y input.
 * @param x_shape        Shape of x input.
 * @param broadcast_spec Broadcast specification mode.
 */
template <class T>
void atan2(const T* y,
           const T* x,
           T* out,
           const Shape& y_shape,
           const Shape& x_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(y, x, out, y_shape, x_shape, broadcast_spec, [](const T y_val, const T x_val) {
        return func::atan2(y_val, x_val);
    });
}

}  // namespace reference
}  // namespace ov
