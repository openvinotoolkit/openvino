//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/ngraph_visibility.hpp"

#define ROUND_MODE_TO_NEAREST_EVEN

namespace ngraph
{
    class NGRAPH_API float16
    {
    public:
        constexpr float16()
            : m_value{0}
        {
        }

        static uint32_t constexpr frac_size = 10;
        static uint32_t constexpr exp_size = 5;
        static uint32_t constexpr exp_bias = 15;

        float16(uint32_t sign, uint32_t biased_exponent, uint32_t fraction)
            : m_value((sign & 0x01) << 15 | (biased_exponent & 0x1F) << 10 | (fraction & 0x03FF))
        {
        }

        float16(float value);

        std::string to_string() const;
        size_t size() const;
        bool operator==(const float16& other) const;
        bool operator!=(const float16& other) const { return !(*this == other); }
        bool operator<(const float16& other) const;
        bool operator<=(const float16& other) const;
        bool operator>(const float16& other) const;
        bool operator>=(const float16& other) const;
        operator float() const;

        static constexpr float16 from_bits(uint16_t bits) { return float16(bits, true); }
        uint16_t to_bits() const;
        friend std::ostream& operator<<(std::ostream& out, const float16& obj)
        {
            out << static_cast<float>(obj);
            return out;
        }

    private:
        constexpr float16(uint16_t x, bool)
            : m_value{x}
        {
        }
        union F32 {
            F32(float val)
                : f{val}
            {
            }
            F32(uint32_t val)
                : i{val}
            {
            }
            float f;
            uint32_t i;
        };

        uint16_t m_value;
    };
}

namespace std
{
    bool NGRAPH_API isnan(ngraph::float16 x);

    template <>
    class numeric_limits<ngraph::float16>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr ngraph::float16 min() noexcept
        {
            return ngraph::float16::from_bits(0x0200);
        }
        static constexpr ngraph::float16 max() noexcept
        {
            return ngraph::float16::from_bits(0x7BFF);
        }
        static constexpr ngraph::float16 lowest() noexcept
        {
            return ngraph::float16::from_bits(0xFBFF);
        }
        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr int radix = 2;
        static constexpr ngraph::float16 epsilon() noexcept
        {
            return ngraph::float16::from_bits(0x1200);
        }
        static constexpr ngraph::float16 round_error() noexcept
        {
            return ngraph::float16::from_bits(0x3C00);
        }
        static constexpr int min_exponent = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent = 16;
        static constexpr int max_exponent10 = 4;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr float_denorm_style has_denorm = denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr ngraph::float16 infinity() noexcept
        {
            return ngraph::float16::from_bits(0x7C00);
        }
        static constexpr ngraph::float16 quiet_NaN() noexcept
        {
            return ngraph::float16::from_bits(0x7FFF);
        }
        static constexpr ngraph::float16 signaling_NaN() noexcept
        {
            return ngraph::float16::from_bits(0x7DFF);
        }
        static constexpr ngraph::float16 denorm_min() noexcept
        {
            return ngraph::float16::from_bits(0);
        }
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = false;
        static constexpr bool is_modulo = false;
        static constexpr bool traps = false;
        static constexpr bool tinyness_before = false;
        static constexpr float_round_style round_style = round_to_nearest;
    };
}
