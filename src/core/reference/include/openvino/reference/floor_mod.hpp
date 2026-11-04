// Copyright (C) 2018-2026 Intel Corporation
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

template <class T, typename std::enable_if<std::is_signed<T>::value && !ov::is_floating_point<T>()>::type* = nullptr>
T floor_mod(const T x, const T y) {
    // Cast to double is needed for integer input (signed),
    // otherwise std::floor will act like std::trunc
    const double divisor = static_cast<double>(y);
    return static_cast<T>(x - y * std::floor(x / divisor));
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T floor_mod(const T x, const T y) {
    // From the exact std::fmod: x - y * floor(x / y) gave 0 when x / y was beyond double precision, for example
    // 2^80 and 3. The result has the sign of y, zero included.
    const auto r = std::fmod(x, y);  // float for f16 and bf16
    const auto d = static_cast<decltype(r)>(y);
    if (r == 0) {
        return static_cast<T>(std::copysign(decltype(r){0}, d));
    }
    return static_cast<T>(((r < 0) != (d < 0)) ? r + d : r);
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
