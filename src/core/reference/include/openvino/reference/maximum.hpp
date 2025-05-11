// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T>
T max(const T a, const T b) {
    return std::max(a, b);
}
}  // namespace func

template <typename T>
void maximum(const T* arg0, const T* arg1, T* out, size_t count) {
    std::transform(arg0, std::next(arg0, count), arg1, out, func::max<T>);
}

/**
 * @brief Reference implementation of binary elementwise Maximum operator.
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg0_shape      Input 0 shape.
 * @param arg1_shape      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T>
void maximum(const T* arg0,
             const T* arg1,
             T* out,
             const Shape& arg0_shape,
             const Shape& arg1_shape,
             const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::max<T>);
}
}  // namespace reference
}  // namespace ov
