// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e8m0.hpp"

#include <cmath>
#include <limits>

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

union f32_t {
    float value;
    uint32_t bits;
};

constexpr uint8_t f32_mantissa_bits{23u};
constexpr uint8_t byte_mask{0xffu};
constexpr uint32_t f32_exponent_bits_mask{0x7f800000u};
constexpr uint32_t f32_mantissa_bits_mask{0x007fffffu};
constexpr uint32_t round_even{0x00400000u};

uint8_t f32_to_f8e8m0_bits(const float value) {
    const auto input = f32_t{value};

    const auto input_exponent_bits = static_cast<uint8_t>((input.bits & f32_exponent_bits_mask) >> f32_mantissa_bits);

    if (value <= 0.0) {
        return 0b00000000;
    } else if (std::isinf(value) || input_exponent_bits == 0b11111110) {
        return 0b11111110;
    } else if (std::isnan(value)) {
        return 0b11111111;
    }

    if ((input.bits & f32_mantissa_bits_mask) > round_even) {
        return input_exponent_bits + 1;
    } else if ((input.bits & f32_mantissa_bits_mask) == round_even) {
        return input_exponent_bits + (input_exponent_bits & 0x1);
    } else {
        return input_exponent_bits;
    }
}
}  // namespace

float8_e8m0::float8_e8m0(const float value) : m_value(f32_to_f8e8m0_bits(value)){};

float8_e8m0::operator float() const {
    if (to_bits() == 0xff) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if (to_bits() == 0x00) {
        return std::numeric_limits<float>::min() / 2;
    }

    return f32_t{.bits = static_cast<uint32_t>(m_value & byte_mask) << f32_mantissa_bits}.value;
}

uint8_t float8_e8m0::to_bits() const {
    return m_value;
}
}  // namespace ov
