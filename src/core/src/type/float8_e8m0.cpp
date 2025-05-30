// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e8m0.hpp"

#include <cmath>
#include <limits>

#include "openvino/core/type/float_util.hpp"
#include "openvino/reference/fake_convert.hpp"

namespace ov {
static_assert(sizeof(ov::float8_e8m0) == 1, "class f8e8m0 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<ov::float8_e8m0, ov::float8_e8m0>::value,
              "should be trivially constructible");
static_assert(std::is_trivially_copyable<ov::float8_e8m0>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<ov::float8_e8m0>::value, "must be trivially destructible");
static_assert(std::numeric_limits<ov::float8_e8m0>::is_specialized, "numeric_limits must be specialized");
static_assert(!std::numeric_limits<ov::float8_e8m0>::is_integer, "numeric_limits::is_integer must be false");

namespace {

constexpr uint8_t f32_mantissa_bits{23u};
constexpr uint32_t f32_exponent_bits_mask{0x7f800000u};
constexpr uint32_t f32_mantissa_bits_mask{0x007fffffu};
constexpr uint32_t round_even{0x00400000u};

uint8_t f32_to_f8e8m0_bits(const float value) {
    const auto input = util::f32_to_u32_bits(value);
    const auto input_exponent_bits = static_cast<uint8_t>((input & f32_exponent_bits_mask) >> f32_mantissa_bits);

    if (std::signbit(value)) {
        return 0b00000000;
    } else if (input_exponent_bits >= 0b11111110) {
        return input_exponent_bits - static_cast<uint8_t>(std::isinf(value));
    } else {
        // normal values
        const auto input_mantissa_bits = input & f32_mantissa_bits_mask;
        return input_exponent_bits + static_cast<uint8_t>((input_mantissa_bits > round_even) ||    // round to nearest
                                                          ((input_mantissa_bits == round_even) &&  // round to even
                                                           (input_exponent_bits & 0x1)));
    }
}
}  // namespace

float8_e8m0::float8_e8m0(const float value) : m_value(f32_to_f8e8m0_bits(value)){};

float8_e8m0::operator float() const {
    constexpr auto f8e8m0_2_power_negative_127 = std::numeric_limits<ov::float8_e8m0>::min();
    constexpr auto float_2_power_negative_127 = std::numeric_limits<float>::min() / 2;

    if (to_bits() == std::numeric_limits<ov::float8_e8m0>::quiet_NaN().to_bits()) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if (to_bits() == f8e8m0_2_power_negative_127.to_bits()) {
        return float_2_power_negative_127;
    } else {
        return util::u32_bits_to_f32(m_value << f32_mantissa_bits);
    }
}

uint8_t float8_e8m0::to_bits() const {
    return m_value;
}
}  // namespace ov
