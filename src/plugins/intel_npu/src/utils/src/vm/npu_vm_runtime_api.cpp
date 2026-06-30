// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"

#include <algorithm>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {

namespace {
constexpr std::string_view NEW_MLIR_RUNTIME_NAME = "openvino_intel_npu_mlir_runtime";
constexpr std::string_view OLD_MLIR_RUNTIME_NAME = "npu_mlir_runtime";
constexpr std::string_view NEW_VM_RUNTIME_NAME = "openvino_intel_npu_vm_runtime";
constexpr std::string_view OLD_VM_RUNTIME_NAME = "npu_interpreter_runtime";

std::string g_libName{NEW_MLIR_RUNTIME_NAME};
bool g_instanceCreated{false};
}  // namespace

NPUVMRuntimeApi::NPUVMRuntimeApi(std::string_view libName) {
    const std::string_view baseName = libName.empty() ? NEW_MLIR_RUNTIME_NAME : libName;
    try {
        auto libPath =
            ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), std::string(baseName) + OV_BUILD_POSTFIX);
        this->lib = ov::util::load_shared_object(libPath);
    } catch (const std::runtime_error& error) {
        // Temporary compatibility for packages built before the runtime library rename.
        const std::string_view fallbackName = baseName == NEW_MLIR_RUNTIME_NAME ? OLD_MLIR_RUNTIME_NAME
                                              : baseName == NEW_VM_RUNTIME_NAME ? OLD_VM_RUNTIME_NAME
                                                                                : std::string_view{};
        if (fallbackName.empty()) {
            OPENVINO_THROW(error.what());
        }

        try {
            auto fallbackLibPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(),
                                                                      std::string(fallbackName) + OV_BUILD_POSTFIX);
            this->lib = ov::util::load_shared_object(fallbackLibPath);
        } catch (const std::runtime_error& fallbackError) {
            OPENVINO_THROW(error.what(), "; fallback failed: ", fallbackError.what());
        }
    }

    try {
#define nmr_symbol_statement(symbol) \
    this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol));
        nmr_symbols_list();
#undef nmr_symbol_statement
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW(error.what());
    }

#define nmr_symbol_statement(symbol)                                                              \
    try {                                                                                         \
        this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol)); \
    } catch (const std::runtime_error&) {                                                         \
        this->symbol = nullptr;                                                                   \
    }
#undef nmr_symbol_statement

#define nmr_symbol_statement(symbol) symbol = this->symbol;
    nmr_symbols_list();
#undef nmr_symbol_statement
}

void NPUVMRuntimeApi::initializeFromBlob(const void* data, size_t size) {
    const size_t headerSize = std::min(size, size_t{20});
    const std::string_view header(static_cast<const char*>(data), headerSize);
    const std::string_view libName =
        (header.find("NPUByte\x00") != std::string_view::npos) ? NEW_VM_RUNTIME_NAME : NEW_MLIR_RUNTIME_NAME;
    initialize(libName);
}

void NPUVMRuntimeApi::initialize(std::string_view libName) {
    const std::string resolvedName{libName.empty() ? NEW_MLIR_RUNTIME_NAME : libName};
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
}

const std::shared_ptr<NPUVMRuntimeApi>& NPUVMRuntimeApi::getInstance() {
    static std::shared_ptr<NPUVMRuntimeApi> instance = std::make_shared<NPUVMRuntimeApi>(g_libName);
    g_instanceCreated = true;
    return instance;
}

}  // namespace intel_npu
