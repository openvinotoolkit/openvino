// Copyright (C) 2018-2024 Intel Corporation
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
 * @brief Class to represent the f4e2m1 type.
 */
class OPENVINO_API float4_e2m1 {
public:
    float4_e2m1() = default;
    float4_e2m1(float value);

    template <typename I>
    explicit float4_e2m1(I value) : float4_e2m1(static_cast<float>(value)) {}

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
    float4_e2m1 operator+(const T& other) const;
    template <typename T>
    float4_e2m1 operator+=(const T& other);
    template <typename T>
    float4_e2m1 operator-(const T& other) const;
    template <typename T>
    float4_e2m1 operator-=(const T& other);
    template <typename T>
    float4_e2m1 operator*(const T& other) const;
    template <typename T>
    float4_e2m1 operator*=(const T& other);
    template <typename T>
    float4_e2m1 operator/(const T& other) const;
    template <typename T>
    float4_e2m1 operator/=(const T& other);

    operator float() const;

    static constexpr float4_e2m1 from_bits(uint8_t bits) {
        return float4_e2m1(bits, true);
    }

    uint8_t to_bits() const;

    friend std::ostream& operator<<(std::ostream& out, const float4_e2m1& obj) {
        out << static_cast<float>(obj);
        return out;
    }

private:
    constexpr float4_e2m1(uint8_t x, bool) : m_value{x} {}

    uint8_t m_value;
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool float4_e2m1::operator==(const T& other) const {
    return (static_cast<float>(*this) == static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool float4_e2m1::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
float4_e2m1 float4_e2m1::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
float4_e2m1 float4_e2m1::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
float4_e2m1 float4_e2m1::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
float4_e2m1 float4_e2m1::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
float4_e2m1 float4_e2m1::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace ov
