// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
constexpr T abs(const T num) {
    return num;
}

template <class T, typename std::enable_if<std::is_signed<T>::value || ov::is_floating_point<T>()>::type* = nullptr>
T abs(const T num) {
    return std::abs(num);
}
}  // namespace func

/**
 * @brief Reference implementation of Abs operator.
 *
 * @param in     Input pointer to data.
 * @param out    Output pointer to results.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void abs(const T* in, T* out, const size_t count) {
    std::transform(in, std::next(in, count), out, &func::abs<T>);
}
}  // namespace reference
}  // namespace ov
