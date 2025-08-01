// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "utils/cpp/bit_cast.hpp"

/**
 * The bfloat16_t class can be used as an arithmetic type. All arithmetic operations goes through conversion to the
 * float data type.
 */

#define BFLOAT16_ROUND_MODE_TO_NEAREST_EVEN

namespace ov::intel_cpu {

class bfloat16_t {
public:
    bfloat16_t() = default;
    bfloat16_t(float value) noexcept
        : m_value{
#if defined BFLOAT16_ROUND_MODE_TO_NEAREST
              round_to_nearest(value)
#elif defined BFLOAT16_ROUND_MODE_TO_NEAREST_EVEN
              round_to_nearest_even(value)
#elif defined BFLOAT16_ROUND_MODE_TRUNCATE
              truncate(value)
#else
#    error \
        "ROUNDING_MODE must be one of BFLOAT16_ROUND_MODE_TO_NEAREST, BFLOAT16_ROUND_MODE_TO_NEAREST_EVEN, or BFLOAT16_ROUND_MODE_TRUNCATE"
#endif
          } {
    }

    operator float() const {
        auto bits = static_cast<uint32_t>(m_value) << 16;
        return ov::intel_cpu::bit_cast<float>(bits);
    }
    static constexpr bfloat16_t from_bits(uint16_t bits) {
        return {bits, true};
    }
    [[nodiscard]] uint16_t to_bits() const {
        return m_value;
    }

    static uint16_t round_to_nearest_even(float x) {
        uint32_t bits = ov::intel_cpu::bit_cast<uint32_t>(x);
        return static_cast<uint16_t>((bits + ((bits & 0x00010000U) >> 1)) >> 16);
    }

    static uint16_t round_to_nearest(float x) {
        uint32_t bits = ov::intel_cpu::bit_cast<uint32_t>(x);
        return static_cast<uint16_t>((bits + 0x8000U) >> 16);
    }

    static uint16_t truncate(float x) {
        uint32_t bits = ov::intel_cpu::bit_cast<uint32_t>(x);
        return static_cast<uint16_t>(bits >> 16);
    }

private:
    constexpr bfloat16_t(uint16_t x, [[maybe_unused]] bool flag) : m_value{x} {}
    uint16_t m_value{};
};

}  // namespace ov::intel_cpu

/**
 * std::numeric_limits overloaded for better compatibility with template metaprogramming.
 * For example, to make the following template work:
 *  template <typename T>
 *  void someFunction() {
 *      ...
 *      T maxValue = std::numeric_limits<T>::max();
 *      ...
 *  }
 */

namespace std {
template <>
class numeric_limits<ov::intel_cpu::bfloat16_t> {
public:
    static constexpr bool is_specialized = true;
    static constexpr ov::intel_cpu::bfloat16_t min() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x007F);
    }
    static constexpr ov::intel_cpu::bfloat16_t max() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x7F7F);
    }
    static constexpr ov::intel_cpu::bfloat16_t lowest() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0xFF7F);
    }
    static constexpr int digits = 7;
    static constexpr int digits10 = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr ov::intel_cpu::bfloat16_t epsilon() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x3C00);
    }
    static constexpr ov::intel_cpu::bfloat16_t round_error() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x3F00);
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
    static constexpr ov::intel_cpu::bfloat16_t infinity() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x7F80);
    }
    static constexpr ov::intel_cpu::bfloat16_t quiet_NaN() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x7FC0);
    }
    static constexpr ov::intel_cpu::bfloat16_t signaling_NaN() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0x7FC0);
    }
    static constexpr ov::intel_cpu::bfloat16_t denorm_min() noexcept {
        return ov::intel_cpu::bfloat16_t::from_bits(0);
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};
}  // namespace std
