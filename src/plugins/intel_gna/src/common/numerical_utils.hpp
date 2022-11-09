// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <algorithm>

namespace ov {
namespace intel_gna {
namespace common {

#define FLOAT_TO_INT8(a) static_cast<int8_t>(((a) < 0)?((a) - 0.5f):((a) + 0.5f))
#define FLOAT_TO_INT16(a) static_cast<int16_t>(((a) < 0)?((a) - 0.5f):((a) + 0.5f))
#define FLOAT_TO_INT32(a) static_cast<int32_t>(((a) < 0)?((a)-0.5f):((a)+0.5f))
#define FLOAT_TO_INT64(a) static_cast<int64_t>(((a) < 0)?((a)-0.5f):((a)+0.5f))

/**
 * @brief Compares two float values and returns if they are equal
 * @param p1 First float value
 * @param p2 Second float value
 * @return Returns true if two float values are equal
 */
inline bool fp32eq(float p1, float p2, float accuracy = 0.00001f) {
    return (std::abs(p1 - p2) <= accuracy * std::min(std::abs(p1), std::abs(p2)));
}

} // namespace common
} // namespace intel_gna
} // namespace ov
