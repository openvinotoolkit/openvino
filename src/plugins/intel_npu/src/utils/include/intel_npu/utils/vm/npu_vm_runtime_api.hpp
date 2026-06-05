// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

#include "intel_npu/runtime/npu_vm_runtime.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

// clang-format off
#define nvm_symbols_list()                                          \
    nvm_symbol_statement(npuVMRuntimeGetAPIVersion)                 \
    nvm_symbol_statement(npuVMRuntimeCreate)                        \
    nvm_symbol_statement(npuVMRuntimeDestroy)                       \
    nvm_symbol_statement(npuVMRuntimeGetMetadata)                   \
    nvm_symbol_statement(npuVMRuntimeExecute)                       \
    nvm_symbol_statement(npuVMRuntimePredictOutputShape)            \
    nvm_symbol_statement(npuVMRuntimeCreateMemRef)                  \
    nvm_symbol_statement(npuVMRuntimeDestroyMemRef)                 \
    nvm_symbol_statement(npuVMRuntimeSetMemRef)                     \
    nvm_symbol_statement(npuVMRuntimeParseMemRef)                   \
    nvm_symbol_statement(npuVMRuntimeCreateExecutionContext)        \
    nvm_symbol_statement(npuVMRuntimeDestroyExecutionContext)       \
    nvm_symbol_statement(npuVMRuntimeUpdateMutableCommandList)

// symbols that may not be supported in older versions
#define nvm_weak_symbols_list()                             \
    nvm_symbol_statement(npuVMRuntimePredictOutputShape2)

// clang-format on

class NPUVMRuntimeApi {
public:
    NPUVMRuntimeApi(std::string_view libName = "npu_mlir_runtime");
    NPUVMRuntimeApi(const NPUVMRuntimeApi& other) = delete;
    NPUVMRuntimeApi(NPUVMRuntimeApi&& other) = delete;
    void operator=(const NPUVMRuntimeApi&) = delete;
    void operator=(NPUVMRuntimeApi&&) = delete;

    ~NPUVMRuntimeApi() = default;

    // Must be called before the first getInstance() invocation.
    // Re-initialization with the same library is a no-op after the singleton has been created.
    // Throws only if re-initialized with a different library after creation.
    static void initialize(std::string_view libName);

    // Inspects the blob header to select the appropriate runtime library and calls initialize().
    // Selects "npu_interpreter_runtime" for NPUByte blobs, "npu_mlir_runtime" otherwise.
    static void initializeFromBlob(const void* data, size_t size);

    static const std::shared_ptr<NPUVMRuntimeApi>& getInstance();

#define nvm_symbol_statement(symbol) decltype(&::symbol) symbol;
    nvm_symbols_list();
    nvm_weak_symbols_list();
#undef nvm_symbol_statement

private:
    std::shared_ptr<void> lib;
};

#define nvm_symbol_statement(symbol)                                                                        \
    template <typename... Args>                                                                             \
    inline typename std::invoke_result<decltype(&::symbol), Args...>::type wrapped_##symbol(Args... args) { \
        const auto& ptr = NPUVMRuntimeApi::getInstance();                                                   \
        if (ptr->symbol == nullptr) {                                                                       \
            OPENVINO_THROW("Unsupported symbol " #symbol);                                                  \
        }                                                                                                   \
        return ptr->symbol(std::forward<Args>(args)...);                                                    \
    }
nvm_symbols_list();
nvm_weak_symbols_list();
#undef nvm_symbol_statement
#define nvm_symbol_statement(symbol) inline decltype(&::symbol) symbol = wrapped_##symbol;
nvm_symbols_list();
nvm_weak_symbols_list();
#undef nvm_symbol_statement
}  // namespace intel_npu
