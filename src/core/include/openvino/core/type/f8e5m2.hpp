// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
class OPENVINO_API f8e5m2 {
public:
    f8e5m2() = default;

    static uint32_t constexpr frac_size = 2;
    static uint32_t constexpr exp_size = 5;
    static uint32_t constexpr exp_bias = 15;
    static uint32_t constexpr max_shift = 7;
    static uint32_t constexpr exp_mask = 0x1F;  // 0001 1111
    static uint32_t constexpr mant_mask = 0x3;  // 0000 0011

    f8e5m2(uint32_t sign, uint32_t biased_exponent, uint32_t fraction)
        : m_value((sign & 0x01) << max_shift | (biased_exponent & exp_mask) << frac_size | (fraction & mant_mask)) {}

    f8e5m2(float value);
    f8e5m2(ov::float16 f16_val);

    template <typename I>
    explicit f8e5m2(I value) : m_value{f8e5m2{static_cast<float>(value)}.m_value} {}

    std::string to_string() const;
    size_t size() const;
    template <typename T>
    bool operator==(const T& other) const;
    template <typename T>
    bool operator!=(const T& other) const {
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
    f8e5m2 operator+(const T& other) const;
    template <typename T>
    f8e5m2 operator+=(const T& other);
    template <typename T>
    f8e5m2 operator-(const T& other) const;
    template <typename T>
    f8e5m2 operator-=(const T& other);
    template <typename T>
    f8e5m2 operator*(const T& other) const;
    template <typename T>
    f8e5m2 operator*=(const T& other);
    template <typename T>
    f8e5m2 operator/(const T& other) const;
    template <typename T>
    f8e5m2 operator/=(const T& other);
    operator float() const;

    static f8e5m2 from_bits(uint8_t bits) {
        return f8e5m2(bits, true);
    }
    uint8_t to_bits() const;
    friend std::ostream& operator<<(std::ostream& out, const f8e5m2& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    f8e5m2(uint8_t x, bool) : m_value{x} {}

    uint8_t m_value = 0;
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool f8e5m2::operator==(const T& other) const {
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
}

template <typename T>
bool f8e5m2::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool f8e5m2::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool f8e5m2::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool f8e5m2::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
f8e5m2 f8e5m2::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
f8e5m2 f8e5m2::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
f8e5m2 f8e5m2::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
f8e5m2 f8e5m2::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
f8e5m2 f8e5m2::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
f8e5m2 f8e5m2::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
f8e5m2 f8e5m2::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
f8e5m2 f8e5m2::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace ov
