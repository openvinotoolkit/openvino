// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {
// Use custom implementation as function instead std::less_equal functor, gives smaller binary size.
// If removed or replace check impact on library binary size.
template <class T>
constexpr bool less_eq(const T lhs, const T rhs) {
    return lhs <= rhs;
}
}  // namespace func

template <typename T>
void less_eq(const T* arg0, const T* arg1, char* out, const size_t count) {
    std::transform(arg0, std::next(arg0, count), arg1, out, func::less_eq<T>);
}

/**
 * @brief Reference implementation of binary elementwise LessEqual operator.
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg0_shape      Input 0 shape.
 * @param arg1_shape      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T, typename U>
void less_eq(const T* arg0,
             const T* arg1,
             U* out,
             const Shape& arg0_shape,
             const Shape& arg1_shape,
             const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::less_eq<T>);
}
}  // namespace reference
}  // namespace ov
