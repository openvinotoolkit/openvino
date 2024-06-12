// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>

#include <stdexcept>

#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

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
    symbol_statement(zeMemFree)
// clang-format on

// TODO: remove static
#define symbol_statement(symbol) static decltype(&::symbol) symbol = nullptr;
symbols_list();
#undef symbol_statement

inline void loadSymbols(std::shared_ptr<void> so) {
#define symbol_statement(symbol) symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(so, #symbol));
    symbols_list()
#undef symbol_statement

#define symbol_statement(symbol)                                         \
    if (symbol == nullptr) {                                             \
        throw std::runtime_error("Failed to load symbol for: " #symbol); \
    }
        symbols_list()
#undef symbols_list
}

inline void loadLibary() {
    const std::string baseName = "ze_loader";
    try {
        auto libpath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        auto libSp = ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
        auto libSp = ov::util::load_shared_object(libpath.c_str());
        loadSymbols(libSp);
#endif
    } catch (const std::runtime_error& error) {
        throw error;
    } catch (const std::exception& error) {
        OPENVINO_THROW("Unexpected error while loading the " + baseName + " library");
    }
}

}  // namespace intel_npu
