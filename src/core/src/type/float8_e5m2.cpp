// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e5m2.hpp"

#include <cmath>
#include <limits>

#include "openvino/reference/fake_convert.hpp"

namespace ov {
static_assert(sizeof(float8_e5m2) == 1, "class f8e5m2 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<float8_e5m2, float8_e5m2>::value, "should be trivially constructible");
static_assert(std::is_trivially_copyable<float8_e5m2>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<float8_e5m2>::value, "must be trivially destructible");
static_assert(std::numeric_limits<float8_e5m2>::is_specialized, "numeric_limits must be specialized");
static_assert(!std::numeric_limits<float8_e5m2>::is_integer, "numeric_limits::is_integer must be false");

namespace {

constexpr uint8_t byte_shift = 8;

constexpr uint8_t f8e5m2_e_size = 5;     // f8e5m2 exponent bit size
constexpr uint8_t f8e5m2_e_mask = 0x7c;  // f8e5m2 exponent bit mask
constexpr uint8_t f8e5m2_m_size = 2;     // f8e5m2 mantissa bits size
constexpr uint8_t f8e5m2_m_mask = 0x03;  // f8e5m2 mantissa bit mask

uint8_t f32_to_f8e5m2_bits(const float value) {
    auto f16 = static_cast<float16>(value);
    reference::func::emulate_f8e5m2_on_fp16(&f16, &f16, 1, false);
    return static_cast<uint8_t>((f16.to_bits() >> byte_shift));
}
}  // namespace

float8_e5m2::float8_e5m2(uint32_t sign, uint32_t biased_exponent, uint32_t fraction)
    : m_value((sign & 0x01) << (f8e5m2_e_size + f8e5m2_m_size) |
              (biased_exponent & (f8e5m2_e_mask >> f8e5m2_m_size)) << f8e5m2_m_size | (fraction & f8e5m2_m_mask)) {}

float8_e5m2::float8_e5m2(const float value) : m_value(f32_to_f8e5m2_bits(value)){};

float8_e5m2::operator float() const {
    return static_cast<float>(float16::from_bits((static_cast<uint16_t>(m_value) << byte_shift)));
}

uint8_t float8_e5m2::to_bits() const {
    return m_value;
}
}  // namespace ov
