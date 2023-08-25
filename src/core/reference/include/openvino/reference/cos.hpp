// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T cos(const T in) {
    return std::cos(in);
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T cos(const T in) {
    return std::round(std::cos(in));
}

/**
 * @brief Reference implementation of Cos operator.
 *
 * @param arg    Input buffer pointer with input data.
 * @param out    Output buffer pointer with results.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void cos(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, &cos<T>);
}
}  // namespace reference
}  // namespace ov
