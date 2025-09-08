// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T atan(const T in) {
    return std::atan(in);
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T atan(const T in) {
    return static_cast<T>(std::round(std::atan(in)));
}
}  // namespace func

/**
 * @brief Reference implementation of Atan operator.
 *
 * @param arg    Input buffer pointer with input data.
 * @param out    Output buffer pointer with results.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void atan(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, &func::atan<T>);
}
}  // namespace reference
}  // namespace ov
