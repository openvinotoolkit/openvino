// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/f8e5m2.hpp"

#include <bitset>
#include <cmath>
#include <iostream>
#include <limits>

#include "openvino/reference/fake_convert.hpp"

using namespace ov;

static_assert(sizeof(f8e5m2) == 1, "class f8e5m2 must be exactly 1 byte");

f8e5m2::f8e5m2(float value) : f8e5m2(static_cast<ov::float16>(value)){};
f8e5m2::f8e5m2(ov::float16 f16_val) {
    reference::func::emulate_f8e5m2_on_fp16(&f16_val, &f16_val, 1);
    m_value = static_cast<uint8_t>((f16_val.to_bits() >> 8));
}

f8e5m2::operator float() const {
    const uint16_t sign_mask = uint16_t(m_value & 0x80) << 8;
    return static_cast<float>(ov::float16::from_bits(sign_mask | (static_cast<uint16_t>(m_value) << 8)));
}

size_t f8e5m2::size() const {
    return sizeof(m_value);
}

uint8_t f8e5m2::to_bits() const {
    return m_value;
}

std::string f8e5m2::to_string() const {
    return std::to_string(static_cast<float>(*this));
}
