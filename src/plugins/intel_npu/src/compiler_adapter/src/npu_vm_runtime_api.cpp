// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_vm_runtime_api.hpp"

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {

NPUVMRuntimeApi::NPUVMRuntimeApi(std::string_view libName) {
    const std::string baseName = libName.empty() ? "npu_mlir_runtime" : std::string(libName);
    try {
        auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
        this->lib = ov::util::load_shared_object(libPath);
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW(error.what());
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
    nmr_weak_symbols_list();
#undef nmr_symbol_statement

#define nmr_symbol_statement(symbol) symbol = this->symbol;
    nmr_symbols_list();
    nmr_weak_symbols_list();
#undef nmr_symbol_statement
}

const std::shared_ptr<NPUVMRuntimeApi>& NPUVMRuntimeApi::getInstance() {
    static std::shared_ptr<NPUVMRuntimeApi> instance = std::make_shared<NPUVMRuntimeApi>();
    return instance;
}

}  // namespace intel_npu
