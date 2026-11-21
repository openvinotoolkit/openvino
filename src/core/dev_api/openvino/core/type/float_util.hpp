// Copyright (C) 2018-2026 Intel Corporation
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

/**
 * @brief Round float value to 5 mantissa bits with round-to-odd.
 *
 * The 18 low bits are cleared and bit 18 is set if any of them was set. As it keeps at least two more bits than
 * f8e4m3 and f8e5m2, rounding the result to f8 with round-to-nearest-even gives the same result as a single rounding
 * of the input value. The result is exact in f16 unless it becomes zero or overflows in f8.
 *
 * @param value  Input float value.
 * @return       The rounded float value.
 */
inline float round_to_odd_for_f8(float value) {
    constexpr uint32_t dropped_bits = 18;  // f32 mantissa bits below the 5 kept ones
    constexpr uint32_t mask = (uint32_t{1} << dropped_bits) - 1;
    const uint32_t bits = f32_to_u32_bits(value);
    return u32_bits_to_f32((bits | ((bits & mask) + mask)) & ~mask);
}
}  // namespace
}  // namespace util
}  // namespace ov
