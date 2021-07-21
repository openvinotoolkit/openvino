// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            throw std::runtime_error(#f " command failed with code " + std::to_string(res_)); \
        } \
    } while (false)


namespace cldnn {
namespace ze {


}  // namespace ze
}  // namespace cldnn
