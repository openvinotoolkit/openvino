// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e4m3.hpp"

#include <array>
#include <cmath>
#include <limits>

#include "openvino/core/type/float_util.hpp"

namespace ov {

static_assert(sizeof(float8_e4m3) == 1, "class f8e4m3 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<float8_e4m3, float8_e4m3>::value, "should be trivially constructible");
static_assert(std::is_trivially_copyable<float8_e4m3>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<float8_e4m3>::value, "must be trivially destructible");
static_assert(std::numeric_limits<float8_e4m3>::is_specialized, "numeric_limits must be specialized");
static_assert(!std::numeric_limits<float8_e4m3>::is_integer, "numeric_limits::is_integer must be false");

namespace {
constexpr auto float_nan = std::numeric_limits<float>::quiet_NaN();
// Lookup table for conversion f8 -> float. The f8 bit value without sign bit (masked 0x7f) is LUT offset.
static constexpr std::array<float, 128> f8_to_float_lut{
    0.0f,      0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f,  0.03515625f,  0.0390625f,  0.04296875f,  0.046875f,  0.05078125f,  0.0546875f,  0.05859375f,
    0.0625f,   0.0703125f,   0.078125f,   0.0859375f,   0.09375f,   0.1015625f,   0.109375f,   0.1171875f,
    0.125f,    0.140625f,    0.15625f,    0.171875f,    0.1875f,    0.203125f,    0.21875f,    0.234375f,
    0.25f,     0.28125f,     0.3125f,     0.34375f,     0.375f,     0.40625f,     0.4375f,     0.46875f,
    0.5f,      0.5625f,      0.625f,      0.6875f,      0.75f,      0.8125f,      0.875f,      0.9375f,
    1.0f,      1.125f,       1.25f,       1.375f,       1.5f,       1.625f,       1.75f,       1.875f,
    2.0f,      2.25f,        2.5f,        2.75f,        3.0f,       3.25f,        3.5f,        3.75f,
    4.0f,      4.5f,         5.0f,        5.5f,         6.0f,       6.5f,         7.0f,        7.5f,
    8.0f,      9.0f,         10.0f,       11.0f,        12.0f,      13.0f,        14.0f,       15.0f,
    16.0f,     18.0f,        20.0f,       22.0f,        24.0f,      26.0f,        28.0f,       30.0f,
    32.0f,     36.0f,        40.0f,       44.0f,        48.0f,      52.0f,        56.0f,       60.0f,
    64.0f,     72.0f,        80.0f,       88.0f,        96.0f,      104.0f,       112.0f,      120.0f,
    128.0f,    144.0f,       160.0f,      176.0f,       192.0f,     208.0f,       224.0f,      240.0f,
    256.0f,    288.0f,       320.0f,      352.0f,       384.0f,     416.0f,       448.0f,      float_nan};

constexpr uint8_t f8e4m3_s_mask = 0x80;  // f8e4m3 sign bit mask
constexpr uint8_t f8e4m3_e_size = 4;     // f8e4m3 exponent bit size
constexpr uint8_t f8e4m3_e_mask = 0x78;  // f8e4m3 exponent bit mask
constexpr uint8_t f8e4m3_e_bias = 7;     // f8e4m3 exponent bias
constexpr uint8_t f8e4m3_e_max = 0x0f;   // f8e4m3 exponent max value
constexpr uint8_t f8e4m3_m_size = 3;     // f8e4m3 mantissa bits size
constexpr uint8_t f8e4m3_m_mask = 0x07;  // f8e4m3 mantissa bit mask

uint8_t f32_to_f8e4m3_bits(const float value) {
    constexpr uint32_t f32_s_mask = 0x80000000;  // f32 sign bit mask
    constexpr uint32_t f32_e_mask = 0x7F800000;  // f32 exponent bits mask
    constexpr uint32_t f32_e_bias = 127;         // f32 exponent bias
    constexpr uint32_t f32_e_size = 8;           // f32 exponent bits size
    constexpr uint32_t f32_m_mask = 0x007fffff;  // f32 mantissa bits mask
    constexpr uint32_t f32_m_size = 23;          // f32 mantissa bits size

    constexpr uint32_t f8_e_mask = f8e4m3_e_mask << three_bytes_shift;  // f8 exponent bits mask (on u32)
    constexpr uint32_t f8_m_mask = f8e4m3_m_mask << three_bytes_shift;  // f8 mantissa bits mask (on u32)
    constexpr uint32_t f8_m_hidden_one_mask = 0x08000000;               // f8 mantissa hidden one bits mask (on u32)

    constexpr uint32_t round_half = 0x01ffffff;  // value for half to even round for f8
    constexpr uint32_t round_norm = 0x007fffff;  // value for normal round for f8
    constexpr uint32_t round_even = 0x00800000;  // value for half to even round for f8
    constexpr uint32_t round_odd = 0x01800000;   // value for an non-half to even round for f8

    const auto input = util::f32_to_u32_bits(value);
    auto f8_bits = static_cast<uint8_t>((input & f32_s_mask) >> three_bytes_shift);

    uint32_t f32_e_field = input & f32_e_mask;

    if (f32_e_field == f32_e_mask) {
        f8_bits |= (f8e4m3_e_mask | f8e4m3_m_mask);
    } else if (f32_e_field != 0) {
        int32_t f8_biased_exp = (f32_e_field >> f32_m_size) - (f32_e_bias - f8e4m3_e_bias);
        uint32_t fractional = (input & f32_m_mask) << (f32_e_size - f8e4m3_e_size);

        // for normalized values round apply rounding change f8 fractional and biased exponent
        if ((fractional & round_half) == round_odd || (fractional & round_norm) != 0) {
            fractional += round_even;
            if (0 != (fractional & f8_e_mask)) {
                fractional &= f8_e_mask;
                ++f8_biased_exp;
            }
        }
        fractional &= f8_m_mask;

        // set exponent and mantissa on f8 bits
        if (f8_biased_exp > f8e4m3_e_max) {
            // Use NAN as this type has no infinity
            f8_bits |= (f8e4m3_e_mask | f8e4m3_m_mask);
        } else if (f8_biased_exp > 0) {
            f8_bits |= (f8_biased_exp << f8e4m3_m_size) | (fractional >> three_bytes_shift);
        } else {
            // Restore the hidden 1 in f8 mantissa for subnormal calculation
            fractional = f8_m_hidden_one_mask | (input & f32_m_mask) << (f32_e_size - f8e4m3_e_size);
            // Will any bits be shifted off?
            int32_t shift = f8_biased_exp < -(f8e4m3_e_max) ? 0 : (1U << (1 - f8_biased_exp));
            uint32_t sticky = (fractional & (shift - 1)) ? 1 : 0;

            fractional = ((1 + f8_biased_exp) > f8e4m3_e_max) ? 0 : fractional >> (1 - f8_biased_exp);
            fractional |= sticky;
            // apply rounding
            if (((fractional & round_half) == round_odd) || ((fractional & round_norm) != 0)) {
                fractional += round_even;
            }

            f8_bits |= fractional >> three_bytes_shift;
        }
    }

    return f8_bits;
}
}  // namespace

float8_e4m3::float8_e4m3(const uint32_t sign, const uint32_t biased_exponent, const uint32_t fraction)
    : m_value(((sign & 0x01U) << (f8e4m3_e_size + f8e4m3_m_size)) |
              (biased_exponent & (f8e4m3_e_mask >> f8e4m3_m_size)) << f8e4m3_m_size | (fraction & f8e4m3_m_mask)) {}

float8_e4m3::float8_e4m3(const float value) : m_value{f32_to_f8e4m3_bits(value)} {}

float8_e4m3::operator float() const {
    auto f32_bits = util::f32_to_u32_bits(f8_to_float_lut[m_value & (f8e4m3_e_mask | f8e4m3_m_mask)]);
    f32_bits |= (m_value & f8e4m3_s_mask) << three_bytes_shift;
    return util::u32_bits_to_f32(f32_bits);
}

uint8_t float8_e4m3::to_bits() const {
    return m_value;
}
}  // namespace ov
