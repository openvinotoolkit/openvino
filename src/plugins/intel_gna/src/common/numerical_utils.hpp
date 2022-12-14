// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdlib>

namespace ov {
namespace intel_gna {
namespace common {

template <typename T>
inline T FloatToInteger(float a) {
    return static_cast<T>((a < 0.0f) ? (a - 0.5f) : (a + 0.5f));
}
inline int8_t FloatToInt8(float a) {
    return FloatToInteger<int8_t>(a);
}
inline int16_t FloatToInt16(float a) {
    return FloatToInteger<int16_t>(a);
}
inline int32_t FloatToInt32(float a) {
    return FloatToInteger<int32_t>(a);
}
inline int64_t FloatToInt64(float a) {
    return FloatToInteger<int64_t>(a);
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
