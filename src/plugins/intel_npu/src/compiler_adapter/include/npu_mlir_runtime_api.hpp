// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/npu_mlir_runtime.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

// clang-format off
#define nmr_symbols_list()                                      \
    nmr_symbol_statement(npuMLIRRuntimeGetAPIVersion)           \
    nmr_symbol_statement(npuMLIRRuntimeCreate)                  \
    nmr_symbol_statement(npuMLIRRuntimeDestroy)                 \
    nmr_symbol_statement(npuMLIRRuntimeGetMetadata)             \
    nmr_symbol_statement(npuMLIRRuntimeExecute)                 \
    nmr_symbol_statement(npuMLIRRuntimePredictOutputShape)      \
    nmr_symbol_statement(npuMLIRRuntimeCreateMemRef)            \
    nmr_symbol_statement(npuMLIRRuntimeDestroyMemRef)           \
    nmr_symbol_statement(npuMLIRRuntimeSetMemRef)               \
    nmr_symbol_statement(npuMLIRRuntimeParseMemRef)


//unsupported symbols with older runtime versions
#define nmr_weak_symbols_list()                                     \
    nmr_symbol_statement(npuMLIRRuntimeCreateExecutionContext)      \
    nmr_symbol_statement(npuMLIRRuntimeDestroyExecutionContext)     \
    nmr_symbol_statement(npuMLIRRuntimeUpdateMutableCommandList)
// clang-format on

class NPUMLIRRuntimeApi {
public:
    NPUMLIRRuntimeApi();
    NPUMLIRRuntimeApi(const NPUMLIRRuntimeApi& other) = delete;
    NPUMLIRRuntimeApi(NPUMLIRRuntimeApi&& other) = delete;
    void operator=(const NPUMLIRRuntimeApi&) = delete;
    void operator=(NPUMLIRRuntimeApi&&) = delete;

    ~NPUMLIRRuntimeApi() = default;

    static const std::shared_ptr<NPUMLIRRuntimeApi>& getInstance();

#define nmr_symbol_statement(symbol) decltype(&::symbol) symbol;
    nmr_symbols_list();
    nmr_weak_symbols_list();
#undef nmr_symbol_statement

private:
    std::shared_ptr<void> lib;
};

#define nmr_symbol_statement(symbol)                                                                        \
    template <typename... Args>                                                                             \
    inline typename std::invoke_result<decltype(&::symbol), Args...>::type wrapped_##symbol(Args... args) { \
        const auto& ptr = NPUMLIRRuntimeApi::getInstance();                                                 \
        if (ptr->symbol == nullptr) {                                                                       \
            OPENVINO_THROW("Unsupported symbol " #symbol);                                                  \
        }                                                                                                   \
        return ptr->symbol(std::forward<Args>(args)...);                                                    \
    }
nmr_symbols_list();
nmr_weak_symbols_list();
#undef nmr_symbol_statement
#define nmr_symbol_statement(symbol) inline decltype(&::symbol) symbol = wrapped_##symbol;
nmr_symbols_list();
nmr_weak_symbols_list();
#undef nmr_symbol_statement
}  // namespace intel_npu
