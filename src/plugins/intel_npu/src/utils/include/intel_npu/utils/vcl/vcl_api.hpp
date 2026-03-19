// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "vcl.h"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/except.hpp"
namespace intel_npu {

// clang-format off
#define vcl_symbols_list()                                  \
    vcl_symbol_statement(vclGetVersion)                     \
    vcl_symbol_statement(vclCompilerCreate)                 \
    vcl_symbol_statement(vclCompilerDestroy)                \
    vcl_symbol_statement(vclCompilerGetProperties)          \
    vcl_symbol_statement(vclQueryNetworkCreate)             \
    vcl_symbol_statement(vclQueryNetwork)                   \
    vcl_symbol_statement(vclQueryNetworkDestroy)            \
    vcl_symbol_statement(vclExecutableCreate)               \
    vcl_symbol_statement(vclExecutableDestroy)              \
    vcl_symbol_statement(vclExecutableGetSerializableBlob)  \
    vcl_symbol_statement(vclProfilingCreate)                \
    vcl_symbol_statement(vclGetDecodedProfilingBuffer)      \
    vcl_symbol_statement(vclProfilingDestroy)               \
    vcl_symbol_statement(vclProfilingGetProperties)         \
    vcl_symbol_statement(vclLogHandleGetString)             \
    vcl_symbol_statement(vclAllocatedExecutableCreate2)     \
    vcl_symbol_statement(vclGetCompilerSupportedOptions)    \
    vcl_symbol_statement(vclGetCompilerIsOptionSupported)   \


// symbols that may not be supported in older versions of vcl
#define vcl_weak_symbols_list()                             \
    vcl_symbol_statement(vclAllocatedExecutableCreateWSOneShot)
// clang-format on

class VCLApi {
public:
    VCLApi();
    VCLApi(const VCLApi& other) = delete;
    VCLApi(VCLApi&& other) = delete;
    void operator=(const VCLApi&) = delete;
    void operator=(VCLApi&&) = delete;

    static const std::shared_ptr<VCLApi> getInstance();
    std::shared_ptr<void> getLibrary() const {
        return lib;
    }

#define vcl_symbol_statement(vcl_symbol) decltype(&::vcl_symbol) vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

private:
    std::shared_ptr<void> lib;
    Logger _logger;
};

#define vcl_symbol_statement(vcl_symbol)                                                                            \
    template <typename... Args>                                                                                     \
    inline typename std::invoke_result<decltype(&::vcl_symbol), Args...>::type wrapped_##vcl_symbol(Args... args) { \
        const auto& ptr = VCLApi::getInstance();                                                                    \
        if (ptr->vcl_symbol == nullptr) {                                                                           \
            OPENVINO_THROW("Unsupported vcl_symbol " #vcl_symbol);                                                  \
        }                                                                                                           \
        return ptr->vcl_symbol(std::forward<Args>(args)...);                                                        \
    }
vcl_symbols_list();
vcl_weak_symbols_list();
#undef vcl_symbol_statement
#define vcl_symbol_statement(vcl_symbol) inline decltype(&::vcl_symbol) vcl_symbol = wrapped_##vcl_symbol;
vcl_symbols_list();
vcl_weak_symbols_list();
#undef vcl_symbol_statement

}  // namespace intel_npu
