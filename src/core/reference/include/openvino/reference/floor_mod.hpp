// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/mod.hpp"

namespace ov {
namespace reference {
namespace func {

template <class T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
constexpr T floor_mod(const T x, const T y) {
    return mod(x, y);
}

template <class T, typename std::enable_if<ov::is_floating_point<T>() || std::is_signed<T>::value>::type* = nullptr>
T floor_mod(const T x, const T y) {
    // Cast to double is needed for integer input (signed),
    // otherwise std::floor will act like std::trunc
    const double divisor = static_cast<double>(y);
    return static_cast<T>(x - y * std::floor(x / divisor));
}
}  // namespace func

template <typename T>
void floor_mod(const T* arg0,
               const T* arg1,
               T* out,
               const Shape& arg0_shape,
               const Shape& arg1_shape,
               const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::floor_mod<T>);
}
}  // namespace reference
}  // namespace ov
