// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <ze_api.h>

#include <limits>
#include <string>


// Expect success of level zero command, throw runtime error otherwise
#define OV_ZE_EXPECT(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            throw std::runtime_error(#f " command failed with code " + std::to_string(res_)); \
        } \
    } while (false)

// Prints warning if level zero command does not return success result
#define OV_ZE_WARN(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            GPU_DEBUG_COUT << ("[Warning] [GPU] " #f " command failed with code " + std::to_string(res_)); \
        } \
    } while (false)

namespace cldnn {
namespace ze {

static constexpr uint64_t endless_wait = std::numeric_limits<uint64_t>::max();
static constexpr ze_module_format_t ze_module_format_oclc = (ze_module_format_t) 3U;

}  // namespace ze
}  // namespace cldnn
