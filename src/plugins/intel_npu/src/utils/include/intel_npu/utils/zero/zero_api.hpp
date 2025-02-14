// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <loader/ze_loader.h>
#include <ze_api.h>

#include <memory>

#include "openvino/core/except.hpp"

#ifndef _WIN32
#    define LIB_ZE_LOADER_SUFFIX ".1"
#endif

namespace intel_npu {

// clang-format off
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
    symbol_statement(zeMemAllocDevice)                        \
    symbol_statement(zeMemAllocHost)                          \
    symbol_statement(zeMemFree)                               \
    symbol_statement(zeMemGetAllocProperties)                 \
    symbol_statement(zelLoaderGetVersions)

//unsupported symbols with older ze_loader versions
#define weak_symbols_list()                                   \
    symbol_statement(zeCommandListGetNextCommandIdExp)        \
    symbol_statement(zeCommandListUpdateMutableCommandsExp)   \
    symbol_statement(zeInitDrivers)
// clang-format on

class ZeroApi {
public:
    ZeroApi();
    ZeroApi(const ZeroApi& other) = delete;
    ZeroApi(ZeroApi&& other) = delete;
    void operator=(const ZeroApi&) = delete;
    void operator=(ZeroApi&&) = delete;

    static const std::shared_ptr<ZeroApi>& getInstance();

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
