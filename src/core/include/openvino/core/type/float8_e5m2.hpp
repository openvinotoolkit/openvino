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

namespace ov {

/**
 * @brief Class to represent the f8e5m2 type.
 */
class OPENVINO_API float8_e5m2 {
public:
    float8_e5m2() = default;
    float8_e5m2(uint32_t sign, uint32_t biased_exponent, uint32_t fraction);
    float8_e5m2(float value);

    template <typename I>
    explicit float8_e5m2(I value) : m_value{float8_e5m2{static_cast<float>(value)}.m_value} {}

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
    float8_e5m2 operator+(const T& other) const;
    template <typename T>
    float8_e5m2 operator+=(const T& other);
    template <typename T>
    float8_e5m2 operator-(const T& other) const;
    template <typename T>
    float8_e5m2 operator-=(const T& other);
    template <typename T>
    float8_e5m2 operator*(const T& other) const;
    template <typename T>
    float8_e5m2 operator*=(const T& other);
    template <typename T>
    float8_e5m2 operator/(const T& other) const;
    template <typename T>
    float8_e5m2 operator/=(const T& other);

    operator float() const;

    static constexpr float8_e5m2 from_bits(uint8_t bits) {
        return float8_e5m2(bits, true);
    }
    uint8_t to_bits() const;
    friend std::ostream& operator<<(std::ostream& out, const float8_e5m2& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    constexpr float8_e5m2(uint8_t x, bool) : m_value{x} {}

    uint8_t m_value;
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool float8_e5m2::operator==(const T& other) const {
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
bool float8_e5m2::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool float8_e5m2::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool float8_e5m2::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool float8_e5m2::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
float8_e5m2 float8_e5m2::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
float8_e5m2 float8_e5m2::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
float8_e5m2 float8_e5m2::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
float8_e5m2 float8_e5m2::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
float8_e5m2 float8_e5m2::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
float8_e5m2 float8_e5m2::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
float8_e5m2 float8_e5m2::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
float8_e5m2 float8_e5m2::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace ov
