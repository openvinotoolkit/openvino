// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>

#define ROUND_MODE_TRUNCATE

#define cu32(x) (F32(x).vint)

namespace MKLDNNPlugin {
class bfloat16 {
public:
    constexpr bfloat16()
        : m_value{0}
    {
    }
    bfloat16(float value) noexcept
            : m_value{
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

    operator float() const {
        return F32{uint32_t(m_value) << 16}.vfloat;
    }
    static constexpr bfloat16 from_bits(uint16_t bits) { return bfloat16(bits, true); }
    uint16_t to_bits() const { return m_value; }

    static inline uint16_t round_to_nearest_even(float x) {
        return static_cast<uint16_t>((cu32(x) + ((cu32(x) & 0x00010000) >> 1)) >> 16);
    }

    static inline uint16_t round_to_nearest(float x) {
        return static_cast<uint16_t>((cu32(x) + 0x8000) >> 16);
    }

    static inline uint16_t truncate(float x) { return static_cast<uint16_t>((cu32(x)) >> 16); }

private:
    constexpr bfloat16(uint16_t x, bool)
            : m_value{x}
    {
    }
    union alignas(16) F32 {
        F32(float val)
                : vfloat{val} {
        }

        F32(uint32_t val)
                : vint{val} {
        }
        float vfloat;
        uint32_t vint;
    };
    uint16_t m_value;
};
} // namespace MKLDNNPlugin

namespace std {
template <>
class numeric_limits<MKLDNNPlugin::bfloat16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr MKLDNNPlugin::bfloat16 min() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x007F);
    }
    static constexpr MKLDNNPlugin::bfloat16 max() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x7F7F);
    }
    static constexpr MKLDNNPlugin::bfloat16 lowest() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0xFF7F);
    }
    static constexpr int digits = 7;
    static constexpr int digits10 = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr MKLDNNPlugin::bfloat16 epsilon() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x3C00);
    }
    static constexpr MKLDNNPlugin::bfloat16 round_error() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x3F00);
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
    static constexpr MKLDNNPlugin::bfloat16 infinity() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x7F80);
    }
    static constexpr MKLDNNPlugin::bfloat16 quiet_NaN() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x7FC0);
    }
    static constexpr MKLDNNPlugin::bfloat16 signaling_NaN() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0x7FC0);
    }
    static constexpr MKLDNNPlugin::bfloat16 denorm_min() noexcept {
        return MKLDNNPlugin::bfloat16::from_bits(0);
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_toward_zero;
};
} // namespace std
