// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T atanh(const T in) {
    return std::atanh(in);
}

// Integral types don't support NAN and INFINITY, use integral limits instead for special values.
template <class T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value>::type* = nullptr>
T atanh(const T in) {
    if (in > 0) {
        return std::numeric_limits<T>::max();
    } else if (in < 0) {
        return std::numeric_limits<T>::min();
    } else {
        return 0;
    }
}

template <class T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
T atanh(const T in) {
    return in > 0 ? std::numeric_limits<T>::max() : 0;
}
}  // namespace func

/**
 * @brief Reference implementation of Atanh operator.
 *
 * @param arg    Input buffer pointer with input data.
 * @param out    Output buffer pointer with results.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void atanh(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, &func::atanh<T>);
}
}  // namespace reference
}  // namespace ov
