// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"

#include <algorithm>
#include <mutex>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {

namespace {
constexpr std::string_view MLIR_RUNTIME_NAME = "openvino_intel_npu_mlir_runtime";
constexpr std::string_view VM_RUNTIME_NAME = "openvino_intel_npu_vm_runtime";

std::string g_libName{MLIR_RUNTIME_NAME};
bool g_instanceCreated{false};
std::mutex g_instanceMutex;
}  // namespace

NPUVMRuntimeApi::NPUVMRuntimeApi(std::string_view libName) {
    const std::string_view baseName = libName.empty() ? MLIR_RUNTIME_NAME : libName;
    try {
        auto libPath =
            ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), std::string(baseName) + OV_BUILD_POSTFIX);
        this->lib = ov::util::load_shared_object(libPath);
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW("Failed to load NPU runtime library '", baseName, "': ", error.what());
    }

    try {
#define nvm_symbol_statement(symbol) \
    this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol));
        nvm_symbols_list();
#undef nvm_symbol_statement
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW("Failed to resolve required symbol in NPU runtime library '", baseName, "': ", error.what());
    }

#define nvm_symbol_statement(symbol)                                                              \
    try {                                                                                         \
        this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol)); \
    } catch (const std::runtime_error&) {                                                         \
        this->symbol = nullptr;                                                                   \
    }
    nvm_weak_symbols_list();
#undef nvm_symbol_statement
}

void NPUVMRuntimeApi::initializeFromBlob(const void* data, size_t size) {
    const size_t headerSize = std::min(size, size_t{20});
    const std::string_view header(static_cast<const char*>(data), headerSize);
    const std::string_view libName =
        (header.find("NPUByte\x00") != std::string_view::npos) ? VM_RUNTIME_NAME : MLIR_RUNTIME_NAME;
    initialize(libName);
}

void NPUVMRuntimeApi::initialize(std::string_view libName) {
    const std::string resolvedName{libName.empty() ? MLIR_RUNTIME_NAME : libName};
    const char* path_env = std::getenv("USE_FIX_PATH");
    if (path_env != nullptr) {
        std::cout << "USE_FIX_PATH is set to: " << path_env << std::endl;
        std::lock_guard<std::mutex> lock(g_instanceMutex);
        std::cout << "Init lock(g_instanceMutex)." << std::endl;
    } else {
        std::cout << "USE_FIX_PATH is not set." << std::endl;
    }
    std::cout << "[3][NPU VM RUNTIME API] g_libName is: " << g_libName << std::endl;
    if (g_instanceCreated) {
        if (g_libName != resolvedName) {
            OPENVINO_THROW("NPUVMRuntimeApi is already initialized with '",
                           g_libName,
                           "', cannot reinitialize with '",
                           resolvedName,
                           "'");
        }
        // Same library — idempotent, nothing to do.
        return;
    }
    g_libName = resolvedName;
    std::cout << "[4][NPU VM RUNTIME API] g_libName is: " << g_libName << std::endl;
}

const std::shared_ptr<NPUVMRuntimeApi>& NPUVMRuntimeApi::getInstance() {
    // Function-local static: initialization is thread-safe in C++11+ and runs exactly once.
    static std::shared_ptr<NPUVMRuntimeApi> instance = []() {
        std::shared_ptr<NPUVMRuntimeApi> runtimeApi;
        const char* path_env = std::getenv("USE_FIX_PATH");
        if (path_env != nullptr) {
            std::cout << "USE_FIX_PATH is set to: " << path_env << std::endl;
            std::lock_guard<std::mutex> lock(g_instanceMutex);
            std::cout << "Init lock(g_instanceMutex)." << std::endl;
            // Explicitly construct with the selected library under the lock.
            runtimeApi = std::make_shared<NPUVMRuntimeApi>(g_libName);
        } else {
            std::cout << "USE_FIX_PATH is not set." << std::endl;
            // Default construction method.
            runtimeApi = std::make_shared<NPUVMRuntimeApi>(g_libName);
            std::cout << "fix compile issue." << std::endl;
        }
        g_instanceCreated = true;
        return runtimeApi;
    }();

    return instance;
}

}  // namespace intel_npu
