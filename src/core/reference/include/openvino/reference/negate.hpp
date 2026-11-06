// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Negative operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
namespace func {
template <class T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value>::type* = nullptr>
constexpr T negate(const T val) {
    using U = typename std::make_unsigned<T>::type;
    return static_cast<T>(U{0} - static_cast<U>(val));
}

template <class T, typename std::enable_if<!(std::is_integral<T>::value && std::is_signed<T>::value)>::type* = nullptr>
constexpr T negate(const T val) {
    return -val;
}
}  // namespace func

template <typename T>
void negate(const T* arg, T* out, const size_t count) {
    std::transform(arg, std::next(arg, count), out, [](const T v) {
        return func::negate(v);
    });
}
}  // namespace reference
}  // namespace ov
