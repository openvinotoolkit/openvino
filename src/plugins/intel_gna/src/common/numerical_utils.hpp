// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace ov {
namespace intel_gna {
namespace common {

template <typename T, typename U, typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
inline T FloatingToInteger(U a) {
    return static_cast<T>((a < 0.0f) ? (a - static_cast<U>(0.5)) : (a + static_cast<U>(0.5)));
}
inline int8_t FloatToInt8(float a) {
    return FloatingToInteger<int8_t>(a);
}
inline int16_t FloatToInt16(float a) {
    return FloatingToInteger<int16_t>(a);
}
inline int32_t FloatToInt32(float a) {
    return FloatingToInteger<int32_t>(a);
}
inline int64_t FloatToInt64(float a) {
    return FloatingToInteger<int64_t>(a);
}
inline int8_t DoubleToInt8(double a) {
    return FloatingToInteger<int8_t>(a);
}
inline int16_t DoubleToInt16(double a) {
    return FloatingToInteger<int16_t>(a);
}
inline int32_t DoubleToInt32(double a) {
    return FloatingToInteger<int32_t>(a);
}
inline int64_t DoubleToInt64(double a) {
    return FloatingToInteger<int64_t>(a);
}

/**
 * @brief Compare two floating point values and return true if they are equal with given accuracy
 * @param p1 First floating point value
 * @param p2 Second floating point value
 * @param accuracy accuracy of comparision
 * @return Returns true if two floating point values are equal
 */
template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
bool AreFpEq(T p1, T p2, T accuracy = std::numeric_limits<T>::epsilon()) {
    return (std::abs(p1 - p2) <= accuracy * std::min(std::abs(p1), std::abs(p2)));
}

}  // namespace common
}  // namespace intel_gna
}  // namespace ov
