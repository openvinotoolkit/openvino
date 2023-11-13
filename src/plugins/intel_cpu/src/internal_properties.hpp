// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @brief Defines how many records can be stored in the CPU runtime parameters cache per CPU runtime parameter type per
 * stream.
 */
static constexpr Property<int32_t, PropertyMutability::RW> cpu_runtime_cache_capacity{"CPU_RUNTIME_CACHE_CAPACITY"};

/**
 * @brief Allow low precision transform
 */
static constexpr Property<bool, PropertyMutability::RW> lp_transforms_mode{"LP_TRANSFORMS_MODE"};

/**
 * @brief Enum to define possible snippets mode hints
 */
enum class SnippetsMode {
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
 * @brief Defines Snippets tokenization mode
 * @param ENABLE - default pipeline
 * @param IGNORE_CALLBACK - disable the Snippets markup transformation and tokenization callback
 * @param DISABLE - turn off the Snippets
 */
static constexpr Property<SnippetsMode, PropertyMutability::RW> snippets_mode{"SNIPPETS_MODE"};

}  // namespace intel_cpu
}  // namespace ov