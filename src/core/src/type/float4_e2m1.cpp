// Copyright (C) 2018-2024 Intel Corporation
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

constexpr uint8_t f4e2m1_e_size = 2;     // f4e2m1 exponent bit size
constexpr uint8_t f4e2m1_e_mask = 0x06;  // f4e2m1 exponent bit mask
constexpr uint8_t f4e2m1_e_bias = 1;     // f4e2m1 exponent bias
constexpr uint8_t f4e2m1_e_max = 0x03;   // f4e2m1 exponent max value
constexpr uint8_t f4e2m1_m_size = 1;     // f4e2m1 mantissa bits size
constexpr uint8_t f4e2m1_m_mask = 0x01;  // f4e2m1 mantissa bit mask

uint8_t f32_to_f4e2m1_bits(const float value) {
    constexpr uint32_t f32_s_mask = 0x80000000;  // f32 sign bit mask
    constexpr uint32_t f32_e_mask = 0x7F800000;  // f32 exponent bits mask
    constexpr uint32_t f32_e_bias = 127;         // f32 exponent bias
    constexpr uint32_t f32_e_size = 8;           // f32 exponent bits size
    constexpr uint32_t f32_m_mask = 0x007fffff;  // f32 mantissa bits mask
    constexpr uint32_t f32_m_size = 23;          // f32 mantissa bits size

    constexpr uint32_t f_e_mask = f4e2m1_e_mask << three_bytes_shift;  // f4 exponent bits mask (on u32)
    constexpr uint32_t f_m_mask = f4e2m1_m_mask << three_bytes_shift;  // f4 mantissa bits mask (on u32)
    constexpr uint32_t f_m_hidden_one_mask = 0x02000000;               // f4 mantissa hidden one bits mask (on u32)

    constexpr uint32_t round_half = 0x01ffffff;  // value for half to even round for f4
    constexpr uint32_t round_norm = 0x07ffffff;  // value for normal round for f4
    constexpr uint32_t round_even = 0x00800000;  // value for half to even round for f4
    constexpr uint32_t round_odd = 0x01800000;   // value for an non-half to even round for f4

    const auto input = util::f32_to_u32_bits(value);
    auto f4_bits = static_cast<uint8_t>((input & f32_s_mask) >> (three_bytes_shift + 4U));

    uint32_t f32_e_field = input & f32_e_mask;

    if (f32_e_field == f32_e_mask) {
        f4_bits |= (f4e2m1_e_mask | f4e2m1_m_mask);
    } else if (f32_e_field != 0) {
        int32_t target_f_biased_exp = (f32_e_field >> f32_m_size) - (f32_e_bias - f4e2m1_e_bias);
        uint32_t fractional = (input & f32_m_mask) << (f32_e_size - f4e2m1_e_size - 4U);

        // for normalized values round apply rounding change target fractional and biased exponent
        if ((fractional & round_half) == round_odd || (fractional & round_norm) != 0) {
            fractional += round_even;
            if (0 != (fractional & f_e_mask)) {
                fractional &= f_e_mask;
                ++target_f_biased_exp;
            }
        }
        fractional &= f_m_mask;

        // set exponent and mantissa on target bits
        if (target_f_biased_exp > f4e2m1_e_max) {
            // Use NAN as this type has no infinity
            f4_bits |= (f4e2m1_e_mask | f4e2m1_m_mask);
        } else if (target_f_biased_exp > 0) {
            f4_bits |= (target_f_biased_exp << f4e2m1_m_size) | (fractional >> (three_bytes_shift));
        } else {
            // Restore the hidden 1 in target mantissa for subnormal calculation
            fractional = f_m_hidden_one_mask | (input & f32_m_mask) << (f32_e_size - f4e2m1_e_size - 4U);
            // Will any bits be shifted off?
            int32_t shift = target_f_biased_exp < -(f4e2m1_e_max) ? 0 : (1U << (1 - target_f_biased_exp));
            uint32_t sticky = (fractional & (shift - 1)) ? 1 : 0;

            fractional = ((1 + target_f_biased_exp) > f4e2m1_e_max - 1) ? 0 : fractional >> (1 - target_f_biased_exp);
            fractional |= sticky;
            // apply rounding
            if (((fractional & round_half) == round_odd) || ((fractional & round_norm) != 0)) {
                fractional += round_even;
            }
            f4_bits |= fractional >> three_bytes_shift;
        }
    }

    return f4_bits;
}
}  // namespace

float4_e2m1::float4_e2m1(const float value) : m_value(f32_to_f4e2m1_bits(value)){};

float4_e2m1::operator float() const {
    return f4e2m1_to_f32_lut[m_value];
}

uint8_t float4_e2m1::to_bits() const {
    return m_value;
}
}  // namespace ov
