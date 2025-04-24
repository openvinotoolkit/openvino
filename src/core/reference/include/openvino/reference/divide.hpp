// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {

template <class T>
constexpr T div(const T x, const T y) {
    return x / y;
}

// NOTE: Execution throws `std::domain_error` if either a non-integral value or an
// out-of-bounds value is detected in the input tensor.
template <class T>
T try_div(const T x, const T y) {
    if (y == 0) {
        throw std::domain_error("integer division by zero");
    }
    return div(x, y);
}

template <class T>
T try_python_div(const T x, const T y) {
    if (y == 0) {
        throw std::domain_error("integer division by zero");
    }

    T quot = div(x, y);
    return (((x < 0) != (y < 0)) && (x % y != 0)) ? quot - T{1} : quot;
}
}  // namespace func

template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type divide(const T* arg0,
                                                                 const T* arg1,
                                                                 T* out,
                                                                 const Shape& arg0_shape,
                                                                 const Shape& arg1_shape,
                                                                 const op::AutoBroadcastSpec& broadcast_spec,
                                                                 bool pythondiv) {
    auto div = pythondiv ? func::try_python_div<T> : func::try_div<T>;
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, div);
}

// In English: return type is void and T must be a standard floating point type, or
// bfloat16, or float16.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value ||
                        std::is_same<T, float16>::value>::type
divide(const T* arg0, const T* arg1, T* out, size_t count, bool) {
    // TODO: Here we do not check for div by zero, so we'll get +-inf here
    // if arg1[i] == 0. Is that the right thing to do? Jury's still out.
    std::transform(arg0, arg0 + count, arg1, out, func::div<T>);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value ||
                        std::is_same<T, float16>::value>::type
divide(const T* arg0,
       const T* arg1,
       T* out,
       const Shape& arg0_shape,
       const Shape& arg1_shape,
       const op::AutoBroadcastSpec& broadcast_spec,
       bool) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::div<T>);
}
}  // namespace reference
}  // namespace ov
