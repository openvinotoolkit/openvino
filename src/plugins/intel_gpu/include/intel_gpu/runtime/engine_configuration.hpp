// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <string_view>

namespace cldnn {

/// @brief Defines available engine types
enum class engine_types : int32_t {
    ocl,
    sycl,
    ze
};

inline std::ostream& operator<<(std::ostream& os, const engine_types& type) {
    switch (type) {
    case engine_types::ocl: os << "ocl"; break;
    case engine_types::sycl: os << "sycl"; break;
    case engine_types::ze: os << "ze"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available runtime types
enum class runtime_types : int32_t {
    ocl,
    sycl,
    ze,
};

inline std::ostream& operator<<(std::ostream& os, const runtime_types& type) {
    switch (type) {
    case runtime_types::ocl: os << "ocl"; break;
    case runtime_types::sycl: os << "sycl"; break;
    case runtime_types::ze: os << "ze"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available backend types
enum class backend_types : int32_t {
    cuda,
    hip,
    ocl,
    ze,
};

inline std::ostream& operator<<(std::ostream& os, const backend_types& type) {
    switch (type) {
    case backend_types::cuda: os << "cuda"; break;
    case backend_types::hip: os << "hip"; break;
    case backend_types::ocl: os << "ocl"; break;
    case backend_types::ze: os << "ze"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Get default engine type
engine_types get_default_engine_type();

/// @brief Get default runtime type
runtime_types get_default_runtime_type();

// Stable string tag for a runtime type ("OCL"/"ZE"/"SYCL"), used to partition on-disk
// caches per runtime. This is a COMPATIBILITY CONSTANT: changing a returned value
// invalidates every existing GPU cache keyed with the old value.
std::string_view to_cache_tag(runtime_types type);

// to_cache_tag(get_default_runtime_type()) for this build.
std::string_view get_runtime_cache_tag();

}  // namespace cldnn
