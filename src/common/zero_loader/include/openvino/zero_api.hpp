// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/loader/ze_loader.h>
#include <level_zero/ze_api.h>

#include <memory>

#include "openvino/core/except.hpp"

namespace ov {

// clang-format off
/**
 * @def symbols_list
 * @brief Macro that expands to declarations of required Level Zero API symbols.
 */
#define symbols_list()                                        \
    symbol_statement(zeCommandListAppendBarrier)              \
    symbol_statement(zeCommandListAppendEventReset)           \
    symbol_statement(zeCommandListAppendMemoryCopy)           \
    symbol_statement(zeCommandListAppendSignalEvent)          \
    symbol_statement(zeCommandListAppendWaitOnEvents)         \
    symbol_statement(zeCommandListAppendWriteGlobalTimestamp) \
    symbol_statement(zeCommandListClose)                      \
    symbol_statement(zeCommandListCreate)                     \
    symbol_statement(zeCommandListDestroy)                    \
    symbol_statement(zeCommandListReset)                      \
    symbol_statement(zeCommandQueueCreate)                    \
    symbol_statement(zeCommandQueueDestroy)                   \
    symbol_statement(zeCommandQueueExecuteCommandLists)       \
    symbol_statement(zeContextCreate)                         \
    symbol_statement(zeContextDestroy)                        \
    symbol_statement(zeDeviceGet)                             \
    symbol_statement(zeDeviceGetCommandQueueGroupProperties)  \
    symbol_statement(zeDeviceGetProperties)                   \
    symbol_statement(zeDevicePciGetPropertiesExt)             \
    symbol_statement(zeDeviceGetExternalMemoryProperties)     \
    symbol_statement(zeDriverGet)                             \
    symbol_statement(zeDriverGetApiVersion)                   \
    symbol_statement(zeDriverGetExtensionFunctionAddress)     \
    symbol_statement(zeDriverGetExtensionProperties)          \
    symbol_statement(zeDriverGetProperties)                   \
    symbol_statement(zeEventCreate)                           \
    symbol_statement(zeEventDestroy)                          \
    symbol_statement(zeEventHostReset)                        \
    symbol_statement(zeEventHostSynchronize)                  \
    symbol_statement(zeEventPoolCreate)                       \
    symbol_statement(zeEventPoolDestroy)                      \
    symbol_statement(zeFenceCreate)                           \
    symbol_statement(zeFenceDestroy)                          \
    symbol_statement(zeFenceHostSynchronize)                  \
    symbol_statement(zeFenceReset)                            \
    symbol_statement(zeInit)                                  \
    symbol_statement(zeMemAllocHost)                          \
    symbol_statement(zeMemFree)                               \
    symbol_statement(zeMemGetAllocProperties)                 \
    symbol_statement(zelLoaderGetVersions)                    \
    symbol_statement(zeModuleBuildLogDestroy)                 \
    symbol_statement(zeModuleDestroy)                         \
    symbol_statement(zeKernelCreate)                          \
    symbol_statement(zeKernelDestroy)                         \
    symbol_statement(zeModuleGetKernelNames)                  \
    symbol_statement(zeModuleGetNativeBinary)                 \
    symbol_statement(zeModuleBuildLogGetString)               \
    symbol_statement(zeEventQueryStatus)                      \
    symbol_statement(zeEventQueryKernelTimestamp)             \
    symbol_statement(zeDeviceGetSubDevices)                   \
    symbol_statement(zeMemAllocShared)                        \
    symbol_statement(zeModuleCreate)                          \
    symbol_statement(zeMemGetAddressRange)                    \
    symbol_statement(zeEventHostSignal)                       \
    symbol_statement(zeMemAllocDevice)                        \
    symbol_statement(zeCommandListHostSynchronize)            \
    symbol_statement(zeCommandListAppendMemoryFill)           \
    symbol_statement(zeDeviceGetComputeProperties)            \
    symbol_statement(zeDeviceGetMemoryProperties)             \
    symbol_statement(zeDeviceGetMemoryAccessProperties)       \
    symbol_statement(zeDeviceGetModuleProperties)             \
    symbol_statement(zeDeviceGetImageProperties)              \
    symbol_statement(zeDeviceGetCacheProperties)              \
    symbol_statement(zeKernelSetArgumentValue)                \
    symbol_statement(zeCommandListCreateImmediate)            \
    symbol_statement(zeKernelSetGroupSize)                    \
    symbol_statement(zeCommandListAppendLaunchKernel)         \
    symbol_statement(zeImageCreate)                           \
    symbol_statement(zeImageDestroy)                          \
    symbol_statement(zeCommandListAppendImageCopy)            \
    symbol_statement(zeCommandListAppendImageCopyFromMemory)  \
    symbol_statement(zeCommandListAppendImageCopyToMemory)

/**
 * @def weak_symbols_list
 * @brief Macro that expands to declarations of optional Level Zero API symbols.
 */
#define weak_symbols_list()                                   \
    symbol_statement(zeCommandListGetNextCommandIdExp)        \
    symbol_statement(zeCommandListUpdateMutableCommandsExp)   \
    symbol_statement(zeInitDrivers)                           \
    symbol_statement(zelGetLoaderVersion)                     \
    symbol_statement(zelSetDriverTeardown)
// clang-format on

/**
 * @class ZeroApi
 * @brief Singleton for dynamically loading and accessing Level Zero API symbols.
 * 
 * Dynamicaly loads ze_loader during construction and resolves required and optional symbols.
 * Provides wrappers for resolved symbols and throws when missing symbol is called.
 * 
 * @note User must store shared pointer returned by get_instance() to prevent unloading.
 */
class ZeroApi {
public:
    ZeroApi();
    ZeroApi(const ZeroApi& other) = delete;
    ZeroApi(ZeroApi&& other) = delete;
    void operator=(const ZeroApi&) = delete;
    void operator=(ZeroApi&&) = delete;

    ~ZeroApi() = default;

    static const std::shared_ptr<ZeroApi> get_instance();

#define symbol_statement(symbol) decltype(&::symbol) symbol;
    symbols_list();
    weak_symbols_list();
#undef symbol_statement

private:
    std::shared_ptr<void> lib;
};

#define symbol_statement(symbol)                                                                            \
    template <typename... Args>                                                                             \
    inline typename std::invoke_result<decltype(&::symbol), Args...>::type wrapped_##symbol(Args... args) { \
        const auto& ptr = ZeroApi::get_instance();                                                           \
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
}  // namespace ov
#ifndef ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#undef symbols_list
#undef weak_symbols_list
#endif
