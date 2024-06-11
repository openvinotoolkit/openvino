// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <climits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {

/**
 * @brief Class to represent the f8e8m0 type.
 */
class OPENVINO_API float8_e8m0 {
    uint8_t f32_to_f8e8m0_bits(const float value) {
        constexpr uint8_t f32_mantissa_bits {23u};
        constexpr uint32_t f32_exponent_bits_mask {0x7f800000u};
        constexpr uint32_t f32_mantissa_bits_mask {0x007fffffu};
        constexpr uint32_t round_even {0x00400000u};

        const auto input_bits = *reinterpret_cast<const uint32_t*>(&value);
        const auto input_exponent_bits = static_cast<uint8_t>((input_bits & f32_exponent_bits_mask) >> f32_mantissa_bits);

        if(value <= 0.0) {
            return 0b00000000;
        }
        else if(std::isinf(value) || input_exponent_bits == 0b11111110) {
            return 0b11111110;
        }
        else if(std::isnan(value)) {
            return 0b11111111;
        }


        if((input_bits & f32_mantissa_bits_mask) > round_even) {
            return  input_exponent_bits + 1;
        }
        else if((input_bits & f32_mantissa_bits_mask) == round_even) {
            return  input_exponent_bits + (input_exponent_bits & 0x1);
        }
        else {
            return input_exponent_bits;
        }

    }

public:
    float8_e8m0() = default;
    // float8_e8m0(uint32_t sign, uint32_t biased_exponent, uint32_t fraction);
    float8_e8m0(const float value) : m_value{f32_to_f8e8m0_bits(value)}{};

    template <typename I>
    explicit float8_e8m0(I value) : m_value{float8_e8m0{static_cast<float>(value)}.m_value} {}

    operator float() const {
        constexpr uint8_t float_mantissa_bits {23u};
        constexpr uint8_t byte_mask {0xffu};
        union {
            uint32_t i_val;
            float f_val;
        };

        if(to_bits() == 0xff) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        else if(to_bits() == 0x00) {
            return std::numeric_limits<float>::min() / 2;
        }

        i_val = static_cast<uint32_t>(m_value & byte_mask) << float_mantissa_bits;
        return f_val;

        // const auto& i_val = static_cast<uint32_t>(m_value) << float_mantissa_bits;
        // return *reinterpret_cast<const float*>(&i_val);
    }

    static constexpr float8_e8m0 from_bits(uint8_t bits) {
        return float8_e8m0(bits, true);
    }
    uint8_t to_bits() const {
        return m_value;
    }
    friend std::ostream& operator<<(std::ostream& out, const float8_e8m0& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    constexpr float8_e8m0(const uint8_t x, bool) : m_value{x} {}

    uint8_t m_value{};
};
