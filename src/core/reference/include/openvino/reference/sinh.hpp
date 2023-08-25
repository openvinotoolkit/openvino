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
T sinh(const T in) {
    return std::sinh(in);
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T sinh(const T in) {
    return std::round(std::sinh(in));
}

/**
 * @brief Reference implementation of Sinh operator.
 *
 * @param arg    Input buffer pointer with input data.
 * @param out    Output buffer pointer with results.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void sinh(const T* arg, T* out, size_t count) {
    std::transform(arg, arg + count, out, &sinh<T>);
}
}  // namespace reference
}  // namespace ov
