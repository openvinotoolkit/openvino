// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_api.hpp"

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
ZeroApi::ZeroApi() {
    const std::string baseName = "ze_loader";
    try {
        auto libpath = ov::util::make_plugin_library_name({}, baseName);
#if !defined(_WIN32) && !defined(ANDROID)
        libpath = libpath + LIB_ZE_LOADER_SUFFIX;
#endif

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        this->lib = ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
        this->lib = ov::util::load_shared_object(libpath.c_str());
#endif
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW(error.what());
    }

    try {
#define symbol_statement(symbol) \
    this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol));
        symbols_list();
#undef symbol_statement
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW(error.what());
    }

#define symbol_statement(symbol)                                                                  \
    try {                                                                                         \
        this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol)); \
    } catch (const std::runtime_error&) {                                                         \
        this->symbol = nullptr;                                                                   \
    }
    weak_symbols_list();
#undef symbol_statement

#define symbol_statement(symbol) symbol = this->symbol;
    symbols_list();
    weak_symbols_list();
#undef symbol_statement
}

const std::shared_ptr<ZeroApi>& ZeroApi::getInstance() {
    static std::shared_ptr<ZeroApi> instance = std::make_shared<ZeroApi>();
    return instance;
}

}  // namespace intel_npu
