// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vcl/vcl_api.hpp"

#include <stdexcept>

#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
VCLLoader::VCLLoader() : _logger("VCLLoader", Logger::global().level()) {
    const auto baseName = "openvino_intel_npu_compiler_loader";
    const auto libraryDir = ov::util::get_ov_lib_path();
    OPENVINO_ASSERT(!libraryDir.empty(),
                    "Cannot load the VCL compiler library: the directory of the OpenVINO runtime library is unknown");

    try {
        const auto libpath = ov::util::make_plugin_library_name(libraryDir, baseName);
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

}  // namespace intel_npu
