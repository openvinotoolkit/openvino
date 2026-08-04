// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>

namespace cldnn {

/// @brief Defines available engine types
enum class engine_types : int32_t {
    vulkan
};

inline std::ostream& operator<<(std::ostream& os, const engine_types& type) {
    switch (type) {
    case engine_types::vulkan: os << "vulkan"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available runtime types
enum class runtime_types : int32_t {
    vulkan,
};

inline std::ostream& operator<<(std::ostream& os, const runtime_types& type) {
    switch (type) {
    case runtime_types::vulkan: os << "vulkan"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available backend types
enum class backend_types : int32_t {
    cuda,
    hip,
    vulkan,
};

inline std::ostream& operator<<(std::ostream& os, const backend_types& type) {
    switch (type) {
    case backend_types::cuda: os << "cuda"; break;
    case backend_types::hip: os << "hip"; break;
    case backend_types::vulkan: os << "vulkan"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Get default engine type
engine_types get_default_engine_type();

/// @brief Get default runtime type
runtime_types get_default_runtime_type();

}  // namespace cldnn
