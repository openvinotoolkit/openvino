// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_api.hpp"

#include <mutex>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
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

const std::shared_ptr<ZeroApi> ZeroApi::getInstance() {
    static std::mutex mutex;
    static std::weak_ptr<ZeroApi> weak_instance;

    std::lock_guard<std::mutex> lock(mutex);
    auto instance = weak_instance.lock();
    if (!instance) {
        instance = std::make_shared<ZeroApi>();
        weak_instance = instance;
    }
    return instance;
}

}  // namespace intel_npu
