// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/mod.hpp"

namespace ov {
namespace reference {
namespace func {

template <class T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
constexpr T floor_mod(const T x, const T y) {
    return mod(x, y);
}

template <class T, typename std::enable_if<std::is_signed<T>::value && std::is_integral<T>::value>::type* = nullptr>
constexpr T floor_mod(const T x, const T y) {
    // Compute the floored remainder with exact integer arithmetic. Going through double (as the
    // floating-point overload does) loses precision for large magnitudes, e.g. values near the
    // int64 limits, so that FloorMod(INT64_MAX, 3) would wrongly yield 0 instead of 1.
    // C++ division and remainder are undefined when the quotient is not representable.
    if (x == std::numeric_limits<T>::min() && y == T{-1}) {
        return T{0};
    }
    const T rem = static_cast<T>(x % y);
    return (rem != T{0} && ((rem < T{0}) != (y < T{0}))) ? static_cast<T>(rem + y) : rem;
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T floor_mod(const T x, const T y) {
    // Cast to double avoids precision loss for reduced-precision floating-point types.
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
