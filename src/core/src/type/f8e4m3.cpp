// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/f8e4m3.hpp"

#include <cmath>
#include <limits>

#include "openvino/reference/fake_convert.hpp"

namespace ov {
static_assert(sizeof(f8e4m3) == 1, "class f8e4m3 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<f8e4m3, f8e4m3>::value == true);
static_assert(std::is_trivially_constructible<f8e4m3, ov::float16>::value == false);
static_assert(std::is_trivially_constructible<f8e4m3, float>::value == false);

namespace {
constexpr auto f16_exp_bias = 15;
constexpr auto f16_mant_size = 10;
constexpr auto f16_half_size = 8;
constexpr uint16_t f16_exp_mask = 0x7C00;
constexpr uint16_t f16_mant_mask = 0x03FF;
constexpr uint16_t f16_sign_mask = 0x8000;

constexpr auto f8e4m3_exp_bias = 7;
constexpr auto f8e4m3_mant_size = 3;
constexpr auto bias_diff = f16_exp_bias - f8e4m3_exp_bias;
constexpr auto mant_diff = f16_mant_size - f8e4m3_mant_size;

constexpr uint8_t f8_sign_mask = 0x80;      // 1 0000 000
constexpr uint8_t f8e4m3_exp_mask = 0x78;   // 0 1111 000
constexpr uint8_t f8e4m3_mant_mask = 0x07;  // 0 0000 111
}  // namespace

f8e4m3::f8e4m3(float value) : f8e4m3(static_cast<float16>(value)){};
f8e4m3::f8e4m3(float16 f16_val) {
    const uint8_t sign = uint16_t(f16_val.to_bits() & f16_sign_mask) >> f16_half_size;

    reference::func::emulate_f8e4m3_on_fp16(&f16_val, &f16_val, 1);
    const uint8_t exp_bits = static_cast<uint8_t>((f16_val.to_bits() & f16_exp_mask) >> f16_mant_size);
    const uint8_t f8e4m3_mantisa_bits = static_cast<uint8_t>((f16_val.to_bits() & f16_mant_mask) >> mant_diff);
    short exp_val = (exp_bits == 0 && f8e4m3_mantisa_bits == 0) ? 0 : (exp_bits - bias_diff);
    m_value = static_cast<uint8_t>(sign | (exp_val << f8e4m3_mant_size) | f8e4m3_mantisa_bits);
}

f8e4m3::operator float() const {
    const uint16_t sign = (static_cast<uint16_t>(m_value & f8_sign_mask) << f16_half_size);
    const uint16_t exp_bits = static_cast<uint16_t>(m_value & f8e4m3_exp_mask) >> f8e4m3_mant_size;
    const uint16_t f16_mantisa_bits = static_cast<uint16_t>(m_value & f8e4m3_mant_mask)
                                      << (f16_mant_size - f8e4m3_mant_size);

    // TODO: Handle subnormal values (exp_bits == 0 && f16_mantisa_bits != 0)
    short exp_val = (exp_bits == 0 && f16_mantisa_bits == 0) ? 0 : (exp_bits + bias_diff);
    const uint16_t f16_bits = static_cast<uint16_t>(sign | (exp_val << f16_mant_size) | f16_mantisa_bits);
    float16 f16_val = float16::from_bits(f16_bits);
    return static_cast<float>(f16_val);
}

std::string f8e4m3::to_string() const {
    return std::to_string(static_cast<float>(*this));
}

size_t f8e4m3::size() const {
    return sizeof(m_value);
}

uint8_t f8e4m3::to_bits() const {
    return m_value;
}
}  // namespace ov
