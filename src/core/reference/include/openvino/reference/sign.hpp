// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
constexpr T sign(const T v) {
    return static_cast<T>(static_cast<bool>(v));
}

template <class T,
          typename std::enable_if<std::is_floating_point<typename std::decay<T>::type>::value ||
                                  std::is_signed<T>::value>::type* = nullptr>
constexpr T sign(const T v) {
    return static_cast<T>(std::isnan(static_cast<float>(v)) ? v : ((T{0} < v) - (v < T{0})));
}

template <class T,
          typename std::enable_if<std::is_same<float16, typename std::decay<T>::type>::value ||
                                  std::is_same<bfloat16, typename std::decay<T>::type>::value>::type* = nullptr>
T sign(const T v) {
    if (std::isnan(static_cast<float>(v)))
        return v;
    else
        return static_cast<T>((T{0} < v) - (v < T{0}));
}
}  // namespace func

/**
 * @brief Reference implementation of Sign operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void sign(const T* arg, T* out, size_t count) {
    std::transform(arg, arg + count, out, func::sign<T>);
}
}  // namespace reference
}  // namespace ov
