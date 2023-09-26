// Copyright (C) 2018-2023 Intel Corporation
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
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
constexpr T mod(const T x, const T y) {
    return x % y;
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T mod(const T x, const T y) {
    return x - (std::trunc(x / y) * y);
}
}  // namespace func

/**
 * @brief Reference implementation of binary elementwise Mod operator.
 *
 * @param arg0            Iterator to input 0 data.
 * @param arg1            Iterator to input 1 data.
 * @param out             Iterator to output data.
 * @param arg_shape0      Input 0 shape.
 * @param arg_shape1      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <class InputIt, class OutputIt>
void mod(InputIt arg0,
         InputIt arg1,
         OutputIt out,
         const Shape& arg_shape0,
         const Shape& arg_shape1,
         const op::AutoBroadcastSpec& broadcast_spec) {
    using T = typename std::iterator_traits<OutputIt>::value_type;
    autobroadcast_binop(arg0, arg1, out, arg_shape0, arg_shape1, broadcast_spec, &func::mod<T>);
}
}  // namespace reference
}  // namespace ov
