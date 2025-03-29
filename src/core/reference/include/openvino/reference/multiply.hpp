// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {
// Usage of custom function instead of std::multiplies gives smaller binary size.
template <class T>
constexpr T multiply(const T a, const T b) {
    return a * b;
}
}  // namespace func

template <typename T>
void multiply(const T* arg0, const T* arg1, T* out, const size_t count) {
    std::transform(arg0, arg0 + count, arg1, out, func::multiply<T>);
}

/**
 * @brief Reference implementation of binary elementwise Multiply operator.
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg_shape0      Input 0 shape.
 * @param arg_shape1      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T>
void multiply(const T* arg0,
              const T* arg1,
              T* out,
              const Shape& arg0_shape,
              const Shape& arg1_shape,
              const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::multiply<T>);
}
}  // namespace reference
}  // namespace ov
