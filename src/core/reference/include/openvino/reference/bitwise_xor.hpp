// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/xor.hpp"

namespace ov {
namespace reference {
namespace func {
// Check for char datatype used by ov::element::boolean
template <class T, typename std::enable_if<std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
constexpr T bit_xor(const T a, const T b) {
    return logical_xor(a, b);
}

template <class T, typename std::enable_if<!std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
constexpr T bit_xor(const T a, const T b) {
    return a ^ b;
}
}  // namespace func

/**
 * @brief Reference implementation of binary elementwise bitwise XOR operator.
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg_shape0      Input 0 shape.
 * @param arg_shape1      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <class T>
// Check for char datatype used by ov::element::boolean
void bitwise_xor(const T* arg0,
                 const T* arg1,
                 T* out,
                 const Shape& arg0_shape,
                 const Shape& arg1_shape,
                 const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::bit_xor<T>);
}
}  // namespace reference
}  // namespace ov
