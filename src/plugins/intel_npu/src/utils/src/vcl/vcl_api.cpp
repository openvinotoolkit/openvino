// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vcl/vcl_api.hpp"

#include <mutex>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
VCLLoader::VCLLoader(const std::string& library_dir) : _logger("VCLLoader", Logger::global().level()) {
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
    _functions.vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol));
        vcl_symbols_list();
#undef vcl_symbol_statement
    } catch (const std::runtime_error& error) {
        _logger.debug("Failed to get formal symbols from %s", baseName);
        OPENVINO_THROW(error.what());
    }

#define vcl_symbol_statement(vcl_symbol)                                                                           \
    try {                                                                                                          \
        _functions.vcl_symbol = reinterpret_cast<decltype(&::vcl_symbol)>(ov::util::get_symbol(lib, #vcl_symbol)); \
    } catch (const std::runtime_error&) {                                                                          \
        _logger.debug("Failed to get %s from %s", #vcl_symbol, baseName);                                          \
        _functions.vcl_symbol = nullptr;                                                                           \
    }
    vcl_weak_symbols_list();
#undef vcl_symbol_statement
}

const std::shared_ptr<const VCLLoader> VCLLoader::getInstance(const std::string& library_dir) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    static std::string initialized_dir;
    static std::shared_ptr<const VCLLoader> instance = nullptr;

    if (!instance) {
        if (library_dir.empty()) {
            OPENVINO_THROW("VCLLoader instance has not been loaded yet, and no valid path was provided to load it.");
        }
        initialized_dir = library_dir;
        // Not make_shared: the loading constructor is private so that this is the only way to load.
        instance = std::shared_ptr<const VCLLoader>(new VCLLoader(library_dir));
    } else {
        if (!library_dir.empty() && library_dir != initialized_dir) {
            OPENVINO_THROW("VCLLoader has already been initialized with path: '",
                           initialized_dir,
                           "'. Dynamic switching to a new compiler path: '",
                           library_dir,
                           "' in the same process is not supported.");
        }
    }

    return instance;
}

}  // namespace intel_npu
