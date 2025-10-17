// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ze_api.h>

#include <limits>

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

static constexpr uint64_t default_timeout = std::numeric_limits<uint64_t>::max();

}  // namespace ze
}  // namespace cldnn
