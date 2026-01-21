// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace cldnn {

/// @brief Defines available engine types
enum class engine_types : int32_t {
    ocl,
    sycl
};

inline std::ostream& operator<<(std::ostream& os, const engine_types& type) {
    switch (type) {
    case engine_types::ocl: os << "ocl"; break;
    case engine_types::sycl: os << "sycl"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available runtime types
enum class runtime_types : int32_t {
    ocl,
    sycl,
};

inline std::ostream& operator<<(std::ostream& os, const runtime_types& type) {
    switch (type) {
    case runtime_types::ocl: os << "ocl"; break;
    case runtime_types::sycl: os << "sycl"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available backend types
enum class backend_types : int32_t {
    cuda,
    hip,
    ocl,
    l0,
};

inline std::ostream& operator<<(std::ostream& os, const backend_types& type) {
    switch (type) {
    case backend_types::cuda: os << "cuda"; break;
    case backend_types::hip: os << "hip"; break;
    case backend_types::ocl: os << "ocl"; break;
    case backend_types::l0: os << "l0"; break;
    default: os << "unknown"; break;
    }

    return os;
}

}  // namespace cldnn
