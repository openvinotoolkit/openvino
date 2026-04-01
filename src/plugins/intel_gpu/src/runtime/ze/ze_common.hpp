// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/except.hpp"
#define ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#include "openvino/zero_api.hpp"

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

inline const ::ov::ZeroApi& get_ze_api_instance() {
    // Load ZeroApi on first call and keep it alive
    static std::shared_ptr<::ov::ZeroApi> ze_api = ::ov::ZeroApi::get_instance();
    OPENVINO_ASSERT(ze_api != nullptr, "Failed to load ze_loader library");
    return *ze_api;
}

// All Level Zero calls should go through this wrapper
#define symbol_statement(symbol)                                                                            \
    template <typename... Args>                                                                             \
    inline typename std::invoke_result<decltype(&::symbol), Args...>::type wrapped_##symbol(Args... args) { \
        const auto& ze_api = get_ze_api_instance();                                                         \
        if (ze_api.symbol == nullptr) {                                                                     \
            OPENVINO_THROW("Unsupported symbol " #symbol);                                                  \
        }                                                                                                   \
        return ze_api.symbol(std::forward<Args>(args)...);                                                  \
    }
symbols_list();
weak_symbols_list();
#undef symbol_statement
#define symbol_statement(symbol) inline decltype(&::symbol) symbol = wrapped_##symbol;
symbols_list();
weak_symbols_list();
#undef symbol_statement

static constexpr uint64_t endless_wait = std::numeric_limits<uint64_t>::max();
static constexpr ze_module_format_t ze_module_format_oclc = (ze_module_format_t) 3U;

}  // namespace ze
}  // namespace cldnn

#undef symbols_list
#undef weak_symbols_list
