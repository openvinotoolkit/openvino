// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov::reference {
namespace func {
template <class T>
bool equal(const T lhs, const T rhs) {
    return lhs == rhs;
}
}  // namespace func

template <typename T>
void equal(const T* arg0, const T* arg1, char* out, size_t count) {
    std::transform(arg0, std::next(arg0, count), arg1, out, std::equal_to<T>());
}

/**
 * @brief Reference implementation of binary elementwise Equal operator.
 *
 * Used for integral types with custom `equal` function (reduce binary size).
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg_shape0      Input 0 shape.
 * @param arg_shape1      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T, typename U, typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
void equal(const T* arg0,
           const T* arg1,
           U* out,
           const Shape& arg0_shape,
           const Shape& arg1_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    if constexpr (sizeof(T) == 1) {
        autobroadcast_binop(reinterpret_cast<const int8_t*>(arg0),
                            reinterpret_cast<const int8_t*>(arg1),
                            out,
                            arg0_shape,
                            arg1_shape,
                            broadcast_spec,
                            func::equal<int8_t>);
    } else if constexpr (sizeof(T) == 2) {
        autobroadcast_binop(reinterpret_cast<const int16_t*>(arg0),
                            reinterpret_cast<const int16_t*>(arg1),
                            out,
                            arg0_shape,
                            arg1_shape,
                            broadcast_spec,
                            func::equal<int16_t>);
    } else if constexpr (sizeof(T) == 4) {
        autobroadcast_binop(reinterpret_cast<const int32_t*>(arg0),
                            reinterpret_cast<const int32_t*>(arg1),
                            out,
                            arg0_shape,
                            arg1_shape,
                            broadcast_spec,
                            func::equal<int32_t>);
    } else if constexpr (sizeof(T) == 8) {
        autobroadcast_binop(reinterpret_cast<const int64_t*>(arg0),
                            reinterpret_cast<const int64_t*>(arg1),
                            out,
                            arg0_shape,
                            arg1_shape,
                            broadcast_spec,
                            func::equal<int64_t>);
    }
}

/**
 * @brief Reference implementation of binary elementwise Equal operator.
 *
 * Used for floating-point types to (avoid warning compare floating point with `==`).
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg_shape0      Input 0 shape.
 * @param arg_shape1      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T, typename U, typename std::enable_if_t<ov::is_floating_point<T>()>* = nullptr>
void equal(const T* arg0,
           const T* arg1,
           U* out,
           const Shape& arg0_shape,
           const Shape& arg1_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, std::equal_to<T>());
}
}  // namespace ov::reference
