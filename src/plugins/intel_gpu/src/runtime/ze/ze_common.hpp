// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <level_zero/ze_api.h>

#include <limits>

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            throw std::runtime_error(#f " command failed with code " + std::to_string(res_)); \
        } \
    } while (false)


namespace cldnn {
namespace ze {

static constexpr uint64_t default_timeout = std::numeric_limits<uint64_t>::max();

void* find_ze_symbol(const char *symbol);

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

}  // namespace ze
}  // namespace cldnn
