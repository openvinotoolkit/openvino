// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::intel_cpu {

/**
 * @brief Defines how many records can be stored in the CPU runtime parameters cache per CPU runtime parameter type per
 * stream.
 */
static constexpr Property<int32_t, PropertyMutability::RW> cpu_runtime_cache_capacity{"CPU_RUNTIME_CACHE_CAPACITY"};

/**
 * @brief Enum to define possible snippets mode hints.
 */
enum class SnippetsMode : uint8_t {
    ENABLE = 0,           //!<  Enable
    IGNORE_CALLBACK = 1,  //!<  Ignore callback
    DISABLE = 2,          //!<  Disable
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SnippetsMode& mode) {
    switch (mode) {
    case SnippetsMode::ENABLE:
        return os << "ENABLE";
    case SnippetsMode::IGNORE_CALLBACK:
        return os << "IGNORE_CALLBACK";
    case SnippetsMode::DISABLE:
        return os << "DISABLE";
    default:
        OPENVINO_THROW("Unsupported snippets mode value");
    }
}

inline std::istream& operator>>(std::istream& is, SnippetsMode& mode) {
    std::string str;
    is >> str;
    if (str == "ENABLE") {
        mode = SnippetsMode::ENABLE;
    } else if (str == "IGNORE_CALLBACK") {
        mode = SnippetsMode::IGNORE_CALLBACK;
    } else if (str == "DISABLE") {
        mode = SnippetsMode::DISABLE;
    } else {
        OPENVINO_THROW("Unsupported snippets mode: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Define tokenization mode for Snippets.
 * @param ENABLE - default pipeline
 * @param IGNORE_CALLBACK - disable the Snippets markup transformation and tokenization callback
 * @param DISABLE - turn off the Snippets
 */
static constexpr Property<SnippetsMode, PropertyMutability::RW> snippets_mode{"SNIPPETS_MODE"};

/**
 * @brief This property used to test accurcay of setting model_distribution_policy to TENSOR_PARALLEL in functional
 * tests.
 */
static constexpr Property<bool, PropertyMutability::RW> enable_tensor_parallel{"ENABLE_TENSOR_PARALLEL"};

/**
 * @brief Define whether to enable sage_attn
 * @param true - enable
 * @param false - disable
 */
static constexpr Property<bool, PropertyMutability::RW> enable_sage_attn{"ENABLE_SAGE_ATTN"};

/**
 * @brief Per-SDPA-layer KV cache compression override (experimental).
 *
 * Positional vector — entry i applies to the i-th SDPA layer in topological
 * order (matches `ov::Model::get_ordered_ops()` traversal). Each entry is a
 * sub-AnyMap that may contain any of: KEY_CACHE_PRECISION, VALUE_CACHE_PRECISION,
 * KEY_CACHE_QUANT_ALG, VALUE_CACHE_QUANT_ALG, KEY_CACHE_GROUP_SIZE,
 * VALUE_CACHE_GROUP_SIZE, KEY_CACHE_QUANT_MODE, VALUE_CACHE_QUANT_MODE.
 * Missing keys inherit from the global plugin config. Empty sub-map = use
 * global defaults for that layer. Vector length must equal the SDPA layer
 * count of the model.
 */
static constexpr Property<std::vector<ov::AnyMap>, PropertyMutability::RW> kv_cache_per_layer_config{
    "KV_CACHE_PER_LAYER_CONFIG"};

}  // namespace ov::intel_cpu
