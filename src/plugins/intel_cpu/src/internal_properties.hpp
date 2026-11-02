// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

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
 * @brief Where Snippets fuses a Softmax whose result reaches an integer MatMul across a quantizer,
 * apply the softmax row sums to that MatMul's dequantized output instead of to the Softmax, so the
 * quantizer sees exp(s - rowmax) rather than the normalized probabilities. The two agree in exact
 * arithmetic but not in quantized arithmetic, and which one a model wants depends on how its
 * Softmax was calibrated, so this is a decision about a model rather than an optimization and it is
 * off unless it is asked for.
 * @param true - defer the normalization past the MatMul
 * @param false - normalize in place (default)
 */
static constexpr Property<bool, PropertyMutability::RW> snippets_defer_softmax_normalization{
    "SNIPPETS_DEFER_SOFTMAX_NORMALIZATION"};

}  // namespace ov::intel_cpu
