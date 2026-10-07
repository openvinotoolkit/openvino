// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

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
 * Ownership is the caller's: whoever constructs a loader decides how long the library stays
 * loaded. The plugin owns one for its lifetime, via VCLLoaderHolder.
 */
class VCLLoader final : public std::enable_shared_from_this<VCLLoader> {
public:
    explicit VCLLoader(const std::string& library_dir);

    VCLLoader(const VCLLoader& other) = delete;
    VCLLoader(VCLLoader&& other) = delete;
    void operator=(const VCLLoader&) = delete;
    void operator=(VCLLoader&&) = delete;

    /**
     * @brief The function table, sharing this loader's lifetime.
     *
     * An aliasing `shared_ptr`, so a holder of the returned table keeps the library loaded without
     * having to know a library exists. This is the only handle on the library handed out: consumers
     * need nothing else to dispatch safely, and none of them can extend the library's lifetime
     * beyond their own use of the table.
     */
    std::shared_ptr<const VCLFunctionTable> sharedFunctions() const {
        return {shared_from_this(), &_functions};
    }

private:
    VCLFunctionTable _functions;
    std::shared_ptr<void> lib;
    Logger _logger;
};

/**
 * @brief How to obtain the VCL entry points, resolved on first use.
 *
 * Lazy by contract: the library must not be loaded until something actually needs to compile or to
 * decode a profiling buffer, so a driver-only flow never pays for it. Tests supply a populated
 * table with no library behind it at all.
 */
using VCLFunctionTableProvider = std::function<std::shared_ptr<const VCLFunctionTable>()>;

/**
 * @brief Owns one lazily loaded VCL compiler library.
 *
 * Loads at most once, on the first call to `functions()`. Intended to be held by the plugin, so
 * that the library is released when the plugin is destroyed rather than at static teardown.
 *
 * @note A failed load is not remembered - a later call tries again. Nothing retries in a tight
 * loop: CompilerAdapterFactory caches "plugin compiler absent" separately.
 */
class VCLLoaderHolder final {
public:
    explicit VCLLoaderHolder(std::string library_dir);

    VCLLoaderHolder(const VCLLoaderHolder&) = delete;
    VCLLoaderHolder& operator=(const VCLLoaderHolder&) = delete;

    /**
     * @brief The entry points, loading the library if this is the first call.
     * @throws ov::Exception when the library cannot be loaded.
     */
    std::shared_ptr<const VCLFunctionTable> functions() const;

private:
    std::string _library_dir;
    mutable std::mutex _mutex;
    mutable std::shared_ptr<const VCLLoader> _loader;
};

}  // namespace intel_npu
