// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ov::intel_cpu::ternary {

constexpr uint8_t raw_mask = 0x3U;
constexpr uint8_t bits_per_value = 2U;

inline uint8_t encode(int8_t value) {
    if (value > 0) {
        return 0x1U;
    }
    if (value < 0) {
        return 0x3U;
    }
    return 0x0U;
}

inline int8_t decode(uint8_t raw) {
    raw &= raw_mask;
    return raw == 0x3U ? static_cast<int8_t>(-1) : static_cast<int8_t>(raw);
}

inline bool is_zero(uint8_t raw) {
    return (raw & raw_mask) == 0U;
}

inline uint8_t encode_from_float(float value, float positive_threshold = 0.25f, float negative_threshold = -0.25f) {
    if (value > positive_threshold) {
        return encode(1);
    }
    if (value < negative_threshold) {
        return encode(-1);
    }
    return encode(0);
}

inline uint8_t mul(uint8_t lhs, uint8_t rhs) {
    lhs &= raw_mask;
    rhs &= raw_mask;
    if (lhs == 0U || rhs == 0U) {
        return 0U;
    }
    const uint8_t sign = (lhs ^ rhs) & 0x2U;
    return sign ? 0x3U : 0x1U;
}

inline uint8_t add(uint8_t lhs, uint8_t rhs) {
    const int sum = static_cast<int>(decode(lhs)) + static_cast<int>(decode(rhs));
    const int clamped = std::max(-1, std::min(1, sum));
    return encode(static_cast<int8_t>(clamped));
}

inline uint8_t apply_decay(uint8_t state, float decay_rate) {
    const float clamped_decay = std::clamp(decay_rate, 0.0f, 1.0f);
    const float decayed = static_cast<float>(decode(state)) * clamped_decay;
    return encode_from_float(decayed);
}

inline int8_t decay_gate(float decay_rate) {
    return decode(encode_from_float(decay_rate));
}

inline uint8_t read(const uint8_t* base, size_t index) {
    const size_t byte_idx = index / 4U;
    const size_t shift = (index % 4U) * bits_per_value;
    const auto packed = base[byte_idx];
    return static_cast<uint8_t>((packed >> shift) & raw_mask);
}

inline void write(uint8_t* base, size_t index, uint8_t raw_value) {
    const size_t byte_idx = index / 4U;
    const size_t shift = (index % 4U) * bits_per_value;
    const auto mask = static_cast<uint8_t>(raw_mask << shift);
    const uint8_t value = static_cast<uint8_t>((raw_value & raw_mask) << shift);
    base[byte_idx] = static_cast<uint8_t>((base[byte_idx] & ~mask) | value);
}

}  // namespace ov::intel_cpu::ternary
