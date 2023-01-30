// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

#include <string>
#include <stdexcept>
#include <thread>
#include <threading/ie_cpu_streams_executor.hpp>

namespace cldnn {

/// @brief Defines available engine types
enum class engine_types : int32_t {
    ocl,
};

inline std::ostream& operator<<(std::ostream& os, engine_types type) {
    switch (type) {
    case engine_types::ocl: os << "ocl"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available runtime types
enum class runtime_types : int32_t {
    ocl,
};

}  // namespace cldnn
