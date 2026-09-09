// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/except.hpp"
#include "vcl.h"
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
    vcl_symbol_statement(vclAllocatedExecutableCreate4)     \
    vcl_symbol_statement(vclExecutableGetCompatibilityString) \
    vcl_symbol_statement(vclGetCompilerSupportedOptions)    \
    vcl_symbol_statement(vclGetCompilerIsOptionSupported)   \
    vcl_symbol_statement(vclAllocatedExecutableCreateWSOneShot2) \


// symbols that may not be supported in older versions of vcl
#define vcl_weak_symbols_list()                             \
    vcl_symbol_statement(vclAllocatedExecutableCreate2)  // clang-format on

/**
 * @brief The VCL entry points, as a plain aggregate.
 *
 * Deliberately holds no library handle and does no loading: it is data, not behaviour. Every entry
 * point defaults to null, so a default-constructed table is a legitimate value - an unpopulated
 * table
 */
struct VCLFunctionTable {
#define vcl_symbol_statement(vcl_symbol) decltype(&::vcl_symbol) vcl_symbol = nullptr;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

    /**
     * @brief True when every non-weak entry point is populated.
     *
     * Weak symbols are excluded on purpose: they are legitimately null when the loaded library
     * predates them. Consumers that need the full table assert on this at construction, so an
     * unpopulated table fails with a diagnosable error instead of dispatching through a null
     * function pointer at the first call. Generated from `vcl_symbols_list()`, so it cannot drift
     * as the list grows.
     */
    bool hasAllRequiredSymbols() const {
#define vcl_symbol_statement(vcl_symbol) \
    if (this->vcl_symbol == nullptr) {   \
        return false;                    \
    }
        vcl_symbols_list();
#undef vcl_symbol_statement
        return true;
    }
};

/**
 * @brief Owns the loaded VCL compiler library and the function table resolved out of it.
 *
 * The loading constructor is private: `getInstance` is the only way to load, so the process cannot
 * end up with two independently dlopen'd copies of the compiler library.
 */
class VCLLoader final : public std::enable_shared_from_this<VCLLoader> {
public:
    VCLLoader(const VCLLoader& other) = delete;
    VCLLoader(VCLLoader&& other) = delete;
    void operator=(const VCLLoader&) = delete;
    void operator=(VCLLoader&&) = delete;

    static const std::shared_ptr<const VCLLoader> getInstance(const std::string& library_dir = std::string());

    /**
     * @brief The function table, sharing this loader's lifetime.
     *
     * An aliasing `shared_ptr`, so a holder of the returned table keeps the library loaded without
     * having to know a library exists.
     */
    std::shared_ptr<const VCLFunctionTable> sharedFunctions() const {
        return {shared_from_this(), &_functions};
    }

    std::shared_ptr<void> getLibrary() const {
        return lib;
    }

private:
    explicit VCLLoader(const std::string& library_dir);

    VCLFunctionTable _functions;
    std::shared_ptr<void> lib;
    Logger _logger;
};

}  // namespace intel_npu
