// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#include "openvino/zero_api.hpp"

#include <mutex>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

#ifndef _WIN32
#    define LIB_ZE_LOADER_SUFFIX ".1"
#endif

namespace ov {
ZeroApi::ZeroApi() {
    const std::filesystem::path baseName = "ze_loader";
    try {
        auto libpath = ov::util::make_plugin_library_name({}, baseName);
#if !defined(_WIN32) && !defined(ANDROID)
        libpath += LIB_ZE_LOADER_SUFFIX;
#endif
        this->lib = ov::util::load_shared_object(libpath);
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

const std::shared_ptr<ZeroApi> ZeroApi::get_instance() {
    static std::mutex mutex;
    static std::shared_ptr<ZeroApi> instance;

    std::lock_guard<std::mutex> lock(mutex);
    if (!instance) {
        instance = std::make_shared<ZeroApi>();
    }
    return instance;
}

}  // namespace ov
