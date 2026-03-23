// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#include <openvino/zero_api.hpp>

namespace intel_npu {
    using ZeroApi = ::ov::zero::ZeroApi;

#define symbol_statement(symbol)                                                                            \
    template <typename... Args>                                                                             \
    inline typename std::invoke_result<decltype(&::symbol), Args...>::type wrapped_##symbol(Args... args) { \
        const auto& ptr = ZeroApi::getInstance();                                                           \
        if (ptr->symbol == nullptr) {                                                                       \
            OPENVINO_THROW("Unsupported symbol " #symbol);                                                  \
        }                                                                                                   \
        return ptr->symbol(std::forward<Args>(args)...);                                                    \
    }
symbols_list();
weak_symbols_list();
#undef symbol_statement
#define symbol_statement(symbol) inline decltype(&::symbol) symbol = wrapped_##symbol;
symbols_list();
weak_symbols_list();
#undef symbol_statement
}  // namespace intel_npu
