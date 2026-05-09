// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vcl/vcl_api.hpp"

#include <mutex>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
VCLApi::VCLApi() : _logger("VCLApi", Logger::global().level()) {
    std::filesystem::path baseName = "openvino_intel_npu_compiler_loader";
    std::filesystem::path libpath{};

    try {
        libpath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName);
        _logger.debug("Try to load %s", baseName.string().c_str());
        this->lib = ov::util::load_shared_object(libpath);
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to load %s: %s", baseName.string().c_str(), error.what());

        // TODO: remove fallback loading logic after all components switch to "openvino_intel_npu_compiler_loader"
        baseName = "openvino_intel_npu_compiler";
        _logger.debug("Trying to load %s", baseName.string().c_str());
        try {
            libpath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName);
            this->lib = ov::util::load_shared_object(libpath);
        } catch (const std::runtime_error& error) {
            _logger.debug("Failed to load a fallback library: %s", baseName.string().c_str());
            OPENVINO_THROW(error.what());
        }
    }

    try {
#define vcl_symbol_statement(vcl_symbol) \
    this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol));
        vcl_symbols_list();
#undef vcl_symbol_statement
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to get formal symbols from %s", baseName.string().c_str());
        OPENVINO_THROW(error.what());
    }

#define vcl_symbol_statement(vcl_symbol)                                                                      \
    try {                                                                                                     \
        this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol)); \
    } catch (const std::runtime_error&) {                                                                     \
        _logger.debug("Failed to get %s from %s", #vcl_symbol, baseName.string().c_str());                    \
        this->vcl_symbol = nullptr;                                                                           \
    }
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

#define vcl_symbol_statement(vcl_symbol) vcl_symbol = this->vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement
}

const std::shared_ptr<VCLApi> VCLApi::getInstance() {
    static std::shared_ptr<VCLApi> instance = std::make_shared<VCLApi>();
    return instance;
}

}  // namespace intel_npu
