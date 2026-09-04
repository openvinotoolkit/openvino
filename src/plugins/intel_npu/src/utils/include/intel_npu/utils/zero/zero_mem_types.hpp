// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace intel_npu {

// Memory purpose classification - matches GPU plugin for comparison
enum class memory_purpose : uint8_t {
    unknown = 0,
    weights,
    kv_cache_key,
    kv_cache_value,
    activation,
    input,
    output,
    intermediate,
    constant,
    internal_buffer
};

inline const char* memory_purpose_to_string(memory_purpose purpose) {
    switch(purpose) {
        case memory_purpose::weights: return "weights";
        case memory_purpose::kv_cache_key: return "kv_cache_key";
        case memory_purpose::kv_cache_value: return "kv_cache_value";
        case memory_purpose::activation: return "activation";
        case memory_purpose::input: return "input";
        case memory_purpose::output: return "output";
        case memory_purpose::intermediate: return "intermediate";
        case memory_purpose::constant: return "constant";
        case memory_purpose::internal_buffer: return "internal_buffer";
        default: return "unknown";
    }
}

}  // namespace intel_npu
