// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string_view>

#include "intel_npu/runtime/npu_vm_runtime.hpp"
#include "openvino/core/except.hpp"

// TODO: to be removed as soon as we are ready to use NPU VM Runtime API
#include "intel_npu/npu_mlir_runtime.hpp"

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
    nmr_symbol_statement(npuMLIRRuntimeUpdateMutableCommandList)    \
    nmr_symbol_statement(npuVMRuntimeGetAPIVersion)                 \
    nmr_symbol_statement(npuVMRuntimeCreate)                        \
    nmr_symbol_statement(npuVMRuntimeDestroy)                       \
    nmr_symbol_statement(npuVMRuntimeGetMetadata)                   \
    nmr_symbol_statement(npuVMRuntimeExecute)                       \
    nmr_symbol_statement(npuVMRuntimePredictOutputShape)            \
    nmr_symbol_statement(npuVMRuntimeCreateMemRef)                  \
    nmr_symbol_statement(npuVMRuntimeDestroyMemRef)                 \
    nmr_symbol_statement(npuVMRuntimeSetMemRef)                     \
    nmr_symbol_statement(npuVMRuntimeParseMemRef)                   \
    nmr_symbol_statement(npuVMRuntimeCreateExecutionContext)        \
    nmr_symbol_statement(npuVMRuntimeDestroyExecutionContext)       \
    nmr_symbol_statement(npuVMRuntimeUpdateMutableCommandList)

// clang-format on

class NPUVMRuntimeApi {
public:
    NPUVMRuntimeApi(std::string_view libName = "npu_mlir_runtime");
    NPUVMRuntimeApi(const NPUVMRuntimeApi& other) = delete;
    NPUVMRuntimeApi(NPUVMRuntimeApi&& other) = delete;
    void operator=(const NPUVMRuntimeApi&) = delete;
    void operator=(NPUVMRuntimeApi&&) = delete;

    ~NPUVMRuntimeApi() = default;

    static const std::shared_ptr<NPUVMRuntimeApi>& getInstance();

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
        const auto& ptr = NPUVMRuntimeApi::getInstance();                                                 \
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
