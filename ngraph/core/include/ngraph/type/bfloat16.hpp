//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
    class NGRAPH_API bfloat16
    {
    public:
        constexpr bfloat16()
            : m_value{0}
        {
        }
        bfloat16(float value)
            : m_value
        {
#if defined ROUND_MODE_TO_NEAREST
            round_to_nearest(value)
#elif defined ROUND_MODE_TO_NEAREST_EVEN
            round_to_nearest_even(value)
#elif defined ROUND_MODE_TRUNCATE
            truncate(value)
#else
#error                                                                                             \
    "ROUNDING_MODE must be one of ROUND_MODE_TO_NEAREST, ROUND_MODE_TO_NEAREST_EVEN, or ROUND_MODE_TRUNCATE"
#endif
        }
        {
        }

        template <typename I>
        explicit bfloat16(I value)
            : m_value{bfloat16{static_cast<float>(value)}.m_value}
        {
        }

        std::string to_string() const;
        size_t size() const;
        template <typename T>
        bool operator==(const T& other) const;
        template <typename T>
        bool operator!=(const T& other) const
        {
            return !(*this == other);
        }
        template <typename T>
        bool operator<(const T& other) const;
        template <typename T>
        bool operator<=(const T& other) const;
        template <typename T>
        bool operator>(const T& other) const;
        template <typename T>
        bool operator>=(const T& other) const;
        template <typename T>
        bfloat16 operator+(const T& other) const;
        template <typename T>
        bfloat16 operator+=(const T& other);
        template <typename T>
        bfloat16 operator-(const T& other) const;
        template <typename T>
        bfloat16 operator-=(const T& other);
        template <typename T>
        bfloat16 operator*(const T& other) const;
        template <typename T>
        bfloat16 operator*=(const T& other);
        template <typename T>
        bfloat16 operator/(const T& other) const;
        template <typename T>
        bfloat16 operator/=(const T& other);
        operator float() const;

        static std::vector<float> to_float_vector(const std::vector<bfloat16>&);
        static std::vector<bfloat16> from_float_vector(const std::vector<float>&);
        static constexpr bfloat16 from_bits(uint16_t bits) { return bfloat16(bits, true); }
        uint16_t to_bits() const;
        friend std::ostream& operator<<(std::ostream& out, const bfloat16& obj)
        {
            out << static_cast<float>(obj);
            return out;
        }

#define cu32(x) (F32(x).i)

        static uint16_t round_to_nearest_even(float x)
        {
            return static_cast<uint16_t>((cu32(x) + ((cu32(x) & 0x00010000) >> 1)) >> 16);
        }

        static uint16_t round_to_nearest(float x)
        {
            return static_cast<uint16_t>((cu32(x) + 0x8000) >> 16);
        }

        static uint16_t truncate(float x) { return static_cast<uint16_t>((cu32(x)) >> 16); }

    private:
        constexpr bfloat16(uint16_t x, bool)
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

    template <typename T>
    bool bfloat16::operator==(const T& other) const
    {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    }

    template <typename T>
    bool bfloat16::operator<(const T& other) const
    {
        return (static_cast<float>(*this) < static_cast<float>(other));
    }

    template <typename T>
    bool bfloat16::operator<=(const T& other) const
    {
        return (static_cast<float>(*this) <= static_cast<float>(other));
    }

    template <typename T>
    bool bfloat16::operator>(const T& other) const
    {
        return (static_cast<float>(*this) > static_cast<float>(other));
    }

    template <typename T>
    bool bfloat16::operator>=(const T& other) const
    {
        return (static_cast<float>(*this) >= static_cast<float>(other));
    }

    template <typename T>
    bfloat16 bfloat16::operator+(const T& other) const
    {
        return {static_cast<float>(*this) + static_cast<float>(other)};
    }

    template <typename T>
    bfloat16 bfloat16::operator+=(const T& other)
    {
        return *this = *this + other;
    }

    template <typename T>
    bfloat16 bfloat16::operator-(const T& other) const
    {
        return {static_cast<float>(*this) - static_cast<float>(other)};
    }

    template <typename T>
    bfloat16 bfloat16::operator-=(const T& other)
    {
        return *this = *this - other;
    }

    template <typename T>
    bfloat16 bfloat16::operator*(const T& other) const
    {
        return {static_cast<float>(*this) * static_cast<float>(other)};
    }

    template <typename T>
    bfloat16 bfloat16::operator*=(const T& other)
    {
        return *this = *this * other;
    }

    template <typename T>
    bfloat16 bfloat16::operator/(const T& other) const
    {
        return {static_cast<float>(*this) / static_cast<float>(other)};
    }

    template <typename T>
    bfloat16 bfloat16::operator/=(const T& other)
    {
        return *this = *this / other;
    }
}

namespace std
{
    template <>
    class numeric_limits<ngraph::bfloat16>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr ngraph::bfloat16 min() noexcept
        {
            return ngraph::bfloat16::from_bits(0x007F);
        }
        static constexpr ngraph::bfloat16 max() noexcept
        {
            return ngraph::bfloat16::from_bits(0x7F7F);
        }
        static constexpr ngraph::bfloat16 lowest() noexcept
        {
            return ngraph::bfloat16::from_bits(0xFF7F);
        }
        static constexpr int digits = 7;
        static constexpr int digits10 = 2;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr int radix = 2;
        static constexpr ngraph::bfloat16 epsilon() noexcept
        {
            return ngraph::bfloat16::from_bits(0x3C00);
        }
        static constexpr ngraph::bfloat16 round_error() noexcept
        {
            return ngraph::bfloat16::from_bits(0x3F00);
        }
        static constexpr int min_exponent = -125;
        static constexpr int min_exponent10 = -37;
        static constexpr int max_exponent = 128;
        static constexpr int max_exponent10 = 38;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr float_denorm_style has_denorm = denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr ngraph::bfloat16 infinity() noexcept
        {
            return ngraph::bfloat16::from_bits(0x7F80);
        }
        static constexpr ngraph::bfloat16 quiet_NaN() noexcept
        {
            return ngraph::bfloat16::from_bits(0x7FC0);
        }
        static constexpr ngraph::bfloat16 signaling_NaN() noexcept
        {
            return ngraph::bfloat16::from_bits(0x7FC0);
        }
        static constexpr ngraph::bfloat16 denorm_min() noexcept
        {
            return ngraph::bfloat16::from_bits(0);
        }
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = false;
        static constexpr bool is_modulo = false;
        static constexpr bool traps = false;
        static constexpr bool tinyness_before = false;
        static constexpr float_round_style round_style = round_to_nearest;
    };
}
