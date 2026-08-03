// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vcl/vcl_api.hpp"

#include <mutex>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
VCLApi::VCLApi(const std::string& library_dir) : _logger("VCLApi", Logger::global().level()) {
    const auto baseName = "openvino_intel_npu_compiler_loader";

    try {
        const auto libpath = ov::util::make_plugin_library_name(std::filesystem::path(library_dir), baseName);
        _logger.debug("Try to load: %s", ov::util::path_to_string(libpath).c_str());
        this->lib = ov::util::load_shared_object(libpath);
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to load %s: %s", baseName, error.what());
        OPENVINO_THROW(error.what());
    }

    try {
#define vcl_symbol_statement(vcl_symbol) \
    this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol));
        vcl_symbols_list();
#undef vcl_symbol_statement
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to get formal symbols from %s", baseName);
        OPENVINO_THROW(error.what());
    }

#define vcl_symbol_statement(vcl_symbol)                                                                      \
    try {                                                                                                     \
        this->vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol)); \
    } catch (const std::runtime_error&) {                                                                     \
        _logger.debug("Failed to get %s from %s", #vcl_symbol, baseName);                                     \
        this->vcl_symbol = nullptr;                                                                           \
    }
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

#define vcl_symbol_statement(vcl_symbol) vcl_symbol = this->vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement
}

const std::shared_ptr<VCLApi> VCLApi::getInstance(const std::string& library_dir) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    static std::string initialized_dir;
    static std::shared_ptr<VCLApi> instance = nullptr;

    if (!instance) {
        if (library_dir.empty()) {
            OPENVINO_THROW("VCLApi instance has not been loaded yet, and no valid path was provided to load it.");
        }
        initialized_dir = library_dir;
        instance = std::make_shared<VCLApi>(library_dir);
    } else {
        if (!library_dir.empty() && library_dir != initialized_dir) {
            OPENVINO_THROW("VCLApi has already been initialized with path: '",
                           initialized_dir,
                           "'. Dynamic switching to a new compiler path: '",
                           library_dir,
                           "' in the same process is not supported.");
        }
    }

    return instance;
}

}  // namespace intel_npu
