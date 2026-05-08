// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Internal helpers shared between model_builder.cpp and the per-component
// split files (model_builder_norm.cpp, model_builder_rope.cpp, ...).
// Not part of the public test_engine API.
//

#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace ov {
namespace test {
namespace npuw {

// Named constants for magic values used throughout model construction.
inline constexpr float kRoPEBaseFrequency = 10000.0f;
inline constexpr float kAttentionMaskPaddingFP16Min = -65504.0f;

/// Deterministic single fill value from tensor name (for scalars, norms, biases).
inline float fill_value_from_name(const std::string& name) {
    size_t h = std::hash<std::string>{}(name);
    return 0.01f + static_cast<float>(h % 100000u) / 100000.0f;  // [0.01, 1.01)
}

/// Deterministic xorshift32 PRNG — produces pseudo-random per-element values
/// that are reproducible from the tensor name alone.
inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint32_t seed_from_name(const std::string& name) {
    // Ensure non-zero seed (xorshift requires it)
    uint32_t s = static_cast<uint32_t>(std::hash<std::string>{}(name));
    return s ? s : 1u;
}

/// KV cache variable id. Missing separator (e.g. "keypresent") is intentional
/// — matches OV's StatefulToStateless pass regex.
inline std::string make_kv_var_id(const std::string& layer,
                                  const std::string& infix,
                                  const std::string& kv_type) {
    return "past_key_values." + layer + infix + kv_type + "present." + layer + infix + kv_type;
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
