// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <ze_api.h>

#include <limits>
#include <string>
#include <sstream>
#include <iomanip>


// Expect success of level zero command, throw runtime error otherwise
#define OV_ZE_EXPECT(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::stringstream s; \
            s << std::hex << res_; \
            throw std::runtime_error(#f " command failed with code " + s.str()); \
        } \
    } while (false)

// Prints warning if level zero command does not return success result
#define OV_ZE_WARN(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::stringstream s; \
            s << std::hex << res_; \
            GPU_DEBUG_INFO << ("[Warning] [GPU] " #f " command failed with code " + s.str()); \
        } \
    } while (false)

namespace cldnn {
namespace ze {

static constexpr uint64_t endless_wait = std::numeric_limits<uint64_t>::max();
static constexpr ze_module_format_t ze_module_format_oclc = (ze_module_format_t) 3U;

}  // namespace ze
}  // namespace cldnn
