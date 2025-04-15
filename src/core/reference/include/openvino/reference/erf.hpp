// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T erf(const T v) {
    return static_cast<T>(std::round(std::erf(v)));
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T erf(const T v) {
    return std::erf(v);
}
}  // namespace func

/**
 * @brief Reference implementation of Erf operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void erf(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, func::erf<T>);
}
}  // namespace reference
}  // namespace ov
