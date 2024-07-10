// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>

namespace ov {

constexpr uint32_t three_bytes_shift = 24;

namespace util {
namespace {

/**
 * @brief Reinterpret float value to 32 bits.
 *
 * @param value  Input float value.
 * @return       The 32-bite value
 */
inline uint32_t f32_to_u32_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/**
 * @brief Reinterpret 32 bits to float value.
 *
 * @param bits  Input bits.
 * @return      The float value.
 */
inline float u32_bits_to_f32(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}
}  // namespace
}  // namespace util
}  // namespace ov
