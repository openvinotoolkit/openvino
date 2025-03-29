// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

namespace ov {
namespace reference {
namespace func {
// Check for char datatype used by ov::element::boolean
template <class T, typename std::enable_if<std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
T bitwise_not(const T in) {
    return static_cast<T>(!in);
}

template <class T, typename std::enable_if<!std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
T bitwise_not(const T in) {
    return static_cast<T>(~in);
}
}  // namespace func
/**
 * @brief Reference implementation of BitwiseNot operator.
 *
 * @param in     Input pointer to data.
 * @param out    Output pointer to results.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void bitwise_not(const T* in, T* out, size_t count) {
    std::transform(in, std::next(in, count), out, &func::bitwise_not<T>);
}
}  // namespace reference
}  // namespace ov
