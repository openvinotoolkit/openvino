// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include <type_traits>
#include <limits>

#include <details/ie_exception.hpp>

namespace vpu {

using fp16_t = short;

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline constexpr T isPowerOfTwo(T val) {
    return (val > 0) && ((val & (val - 1)) == 0);
}

template <size_t align, typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline constexpr T alignVal(T val) {
    static_assert(isPowerOfTwo(align), "isPowerOfTwo(align)");
    return (val + (align - 1)) & ~(align - 1);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline T alignVal(T val, T align) {
    IE_ASSERT(isPowerOfTwo(align));
    return (val + (align - 1)) & ~(align - 1);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline T divUp(T a, T b) {
    IE_ASSERT(b > 0);
    return (a + b - 1) / b;
}

inline bool isFloatEqual(float a, float b) {
    return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}
inline bool isDoubleEqual(double a, double b) {
    return std::fabs(a - b) <= std::numeric_limits<double>::epsilon();
}

}  // namespace vpu
