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
 * @brief Evaluate the exponential of a softmax numerator with a degree-1 polynomial instead of the
 * accurate one, wherever Snippets fuses the softmax. Buys one fused multiply-add in place of a
 * degree-5 polynomial and a compare/blend pair. Off unless asked for, and x86-64 only.
 *
 * What enabling it costs:
 *  - Max relative error 2.98e-2 on the exponential, against the accurate path's 3.9e-7. Normalising
 *    does not cancel it; the worst case on the normalised probabilities is 6.15e-2.
 *  - exp(0) returns 1.0290141, not 1.
 *  - Saturation in place of underflow and overflow, at neither of the accurate path's knees and to
 *    neither of its values: at or below x = -69.3147 it returns 8.1175e-31 rather than zero, and at
 *    or above x = 88.0297 it returns 1.7508e38 rather than +inf. So a masked-out row entry comes
 *    out at 8.1175e-31 where the accurate path gives it exactly zero. The floor is held that far
 *    above FLT_MIN on purpose, so that dividing it by the row sum still yields a normal for rows of
 *    up to 2^25 elements; no input yields an infinity, a denormal or a NaN, and neither does
 *    normalising one.
 * @param true - approximate
 * @param false - accurate
 */
static constexpr Property<bool, PropertyMutability::RW> snippets_approximate_softmax_exp{
    "SNIPPETS_APPROXIMATE_SOFTMAX_EXP"};

}  // namespace ov::intel_cpu
