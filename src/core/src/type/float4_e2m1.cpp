// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float4_e2m1.hpp"

#include <cmath>
#include <limits>

#include "openvino/core/type/float_util.hpp"
#include "openvino/reference/fake_convert.hpp"

namespace ov {
static_assert(sizeof(float4_e2m1) == 1, "type f4e2m1 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<float4_e2m1, float4_e2m1>::value, "should be trivially constructible");
static_assert(std::is_trivially_copyable<float4_e2m1>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<float4_e2m1>::value, "must be trivially destructible");

// clang-format off
static constexpr std::array<float, 16> f4e2m1_to_f32_lut{
    0.0f,   0.5f,
    1.0f,   1.5f,
    2.0f,   3.0f,
    4.0f,   6.0f,
    -0.0f,  -0.5f,
    -1.0f,  -1.5f,
    -2.0f,  -3.0f,
    -4.0f,  -6.0f};
// clang-format on

namespace {

constexpr uint8_t f4e2m1_e_bias = 1;                     // f4e2m1 exponent bias
constexpr uint8_t f4e2m1_e_max = 0x03;                   // f4e2m1 exponent max value
constexpr uint8_t f4e2m1_m_size = 1;                     // f4e2m1 mantissa bits size
constexpr uint32_t f32_m_size = 23;                      // f32 mantissa bits size
constexpr uint32_t f32_e_mask = 0x7F800000;              // f32 exponent bits mask
constexpr uint32_t f32_m_mask = 0x007FFFFF;              // f32 mantissa bits mask
constexpr uint32_t f32_e_bias = 127;                     // f32 exponent bias
constexpr uint32_t f32_m_round_even_down = 0x200000U;    // f32 mantissa round even down
constexpr uint32_t f32_m_round_even_up_126 = 0x400000U;  // f32 mantissa round even up for exponent 126
constexpr uint32_t f32_m_round_even_up = 0x600000U;      // f32 mantissa round even up

// clang-format off
/**
 * @brief Converts a 32-bit float value to its corresponding 4-bit f4e2m1 representation.
 *
 * The f4e2m1 format uses:
 * - 1 sign bit
 * - 2 exponent bits (with bias 1)
 * - 1 mantissa bit
 *
 * The conversion logic is based on the boundaries and encoding rules for f4e2m1:
 * - Specific ranges are mapped to mantissa and exponent combinations according to the format's specification.
 * - Handles positive and negative values, and clamps exponent values to the maximum allowed.
 *
 * Mapping:
 * | Input: abs(value) |  Result
 * |-------------------+-----------------------
 * |     <= 0.25       |  ( sign_bit | 0b000 )
 * |     <  0.75       |  ( sign_bit | 0b001 )
 * |     <= 1.25       |  ( sign_bit | 0b010 )
 * |     <  1.75       |  ( sign_bit | 0b011 )
 * |     <= 2.5        |  ( sign_bit | 0b100 )
 * |     <  3.5        |  ( sign_bit | 0b101 )
 * |     <= 5.0        |  ( sign_bit | 0b110 )
 * |     >  5.0        |  ( sign_bit | 0b111 )
 *
 * Boundary values for f4e2m1:
 * | Value | Hex        |  Binary (32 bits)                   | Sign | Exponent (8 bits) | Mantissa (23 bits)      | Mantissa (Hex)
 * |-------|------------|-------------------------------------|------|-------------------|-------------------------|---------------
 * | 0.25f | 0x3E800000 | 00111110 10000000 00000000 00000000 |  0   | 01111101 (125)    | 00000000000000000000000 | 0x000000
 * | 0.75f | 0x3F400000 | 00111111 01000000 00000000 00000000 |  0   | 01111110 (126)    | 10000000000000000000000 | 0x400000
 * | 1.25f | 0x3FA00000 | 00111111 10100000 00000000 00000000 |  0   | 01111111 (127)    | 01000000000000000000000 | 0x200000
 * | 1.75f | 0x3FE00000 | 00111111 11100000 00000000 00000000 |  0   | 01111111 (127)    | 11000000000000000000000 | 0x600000
 * | 2.5f  | 0x40200000 | 01000000 00100000 00000000 00000000 |  0   | 10000000 (128)    | 01000000000000000000000 | 0x200000
 * | 3.5f  | 0x40600000 | 01000000 01100000 00000000 00000000 |  0   | 10000000 (128)    | 11000000000000000000000 | 0x600000
 * | 5.0f  | 0x40A00000 | 01000000 10100000 00000000 00000000 |  0   | 10000001 (129)    | 01000000000000000000000 | 0x200000
 *
 *
 * @param value The 32-bit float value to convert.
 * @return The 4-bit f4e2m1 representation as a uint8_t.
 */
// clang-format on
uint8_t f32_to_f4e2m1_bits(float value) {
    const uint32_t bits = util::f32_to_u32_bits(value);
    const uint8_t f32_exp = (bits & f32_e_mask) >> f32_m_size;  // Extract exponent
    const uint32_t f32_mantissa = bits & f32_m_mask;            // 23 bits
    int32_t f4e2m1_exp = f32_exp - f32_e_bias + f4e2m1_e_bias;

    /*
        The f4e2m1 exponent mapping:
        f32_exp      | f4e2m1_exp
        -------------+-----------
        125 and less | (0b0000)
        126          | (0b0000)
        127          | (0b0010)
        128          | (0b0100)
        129 and more | (0b0110)
    */
    if (f4e2m1_exp < 0) {
        f4e2m1_exp = 0;
    }
    if (f4e2m1_exp > f4e2m1_e_max) {
        f4e2m1_exp = f4e2m1_e_max;
    }
    f4e2m1_exp <<= f4e2m1_m_size;

    const auto abs_val = std::abs(value);
    const uint8_t f4_sign_bit = std::signbit(value) ? 0b1000 : 0b0000;

    if (abs_val <= 0.25f) {
        return f4_sign_bit;
    }
    // For the performance reason the if else statements are not using direct float comparison
    else if ((f32_exp == 126 && f32_mantissa >= f32_m_round_even_up_126) ||
             (f32_exp == 127 && f32_mantissa >= f32_m_round_even_up) ||
             (f32_exp == 128 && f32_mantissa >= f32_m_round_even_up)) {
        // 0.75f <= abs_val < 1.0f || 1.75f <= abs_val < 2.0f || 3.5f <= abs_val < 4.0f
        return (f4_sign_bit | f4e2m1_exp) + 2;  // mantissa affect exponent bits
    } else if ((f32_exp == 127 || f32_exp == 128 || f32_exp == 129) && (f32_mantissa <= f32_m_round_even_down)) {
        // 1.0f <= abs_val <= 1.25f || 2.0f <= abs_val <= 2.5f || 4.0f <= abs_val <= 5.0f
        return (f4_sign_bit | f4e2m1_exp);
    } else {
        // 0.25f < abs_val < 0.75f || 1.25f < abs_val < 1.75f || 2.5f < abs_val < 3.5f || 5.0f < abs_val
        return (f4_sign_bit | f4e2m1_exp) + 1;
    }
}
}  // namespace

float4_e2m1::float4_e2m1(const float value) : m_value(f32_to_f4e2m1_bits(value)) {};

float4_e2m1::operator float() const {
    return f4e2m1_to_f32_lut[m_value];
}

uint8_t float4_e2m1::to_bits() const {
    return m_value;
}
}  // namespace ov
