// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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
 * Instances must be owned by a `shared_ptr` (`sharedFunctions()` relies on it), so only
 * VCLLoaderHolder creates them. Whoever owns the holder decides how long the library stays loaded.
 */
class VCLLoader final : public std::enable_shared_from_this<VCLLoader> {
public:
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
    friend class VCLLoaderHolder;

    explicit VCLLoader(const std::string& library_dir);

    VCLFunctionTable _functions;
    std::shared_ptr<void> lib;
    Logger _logger;
};

/**
 * @brief Obtains the VCL entry points on demand.
 *
 * Implementations must stay lazy: the library may only load once something needs to compile or to
 * decode a profiling buffer, so a flow that uses only the driver compiler never pays for it.
 */
using VCLFunctionTableProvider = std::function<std::shared_ptr<const VCLFunctionTable>()>;

/**
 * @brief Owns one lazily loaded VCL compiler library.
 *
 * Loads at most once, on the first call to `functions()`. The plugin holds one, so the library is
 * released with the plugin instead of at static teardown.
 *
 * @note A failed load is not remembered, so a later call tries again. CompilerAdapterFactory caches
 * the absence of the plugin compiler, which keeps callers from retrying in a loop.
 */
class VCLLoaderHolder final {
public:
    /**
     * @brief Loads from the directory of the OpenVINO runtime library.
     * @note The directory is resolved on the first load, not here.
     */
    VCLLoaderHolder() = default;

    /**
     * @brief Loads from the given directory.
     */
    explicit VCLLoaderHolder(const std::string& library_dir);

    VCLLoaderHolder(const VCLLoaderHolder&) = delete;
    VCLLoaderHolder& operator=(const VCLLoaderHolder&) = delete;

    /**
     * @brief Returns the entry points, loading the library on the first call.
     *
     * The returned table keeps the library loaded, even after this holder is destroyed.
     * @throws ov::Exception when the library cannot be loaded.
     */
    std::shared_ptr<const VCLFunctionTable> functions() const;

private:
    std::optional<std::string> _library_dir;
    mutable std::mutex _mutex;
    mutable std::shared_ptr<const VCLLoader> _loader;
};

}  // namespace intel_npu
