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
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T sqrt(const T in) {
    return std::sqrt(in);
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T sqrt(const T in) {
    return static_cast<T>(std::round(std::sqrt(in)));
}
}  // namespace func

/**
 * @brief Reference implementation of Sqrt operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void sqrt(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, func::sqrt<T>);
}
}  // namespace reference
}  // namespace ov
