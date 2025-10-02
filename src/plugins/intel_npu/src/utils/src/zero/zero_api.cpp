// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_api.hpp"

#include <stdio.h>
#ifdef _WIN32
#    include <windows.h>
#endif

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
ZeroApi::ZeroApi() {
    const std::string base_name = "ze_loader";
    try {
        auto libpath = ov::util::make_plugin_library_name({}, base_name);
#if !defined(_WIN32) && !defined(ANDROID)
        libpath = libpath + LIB_ZE_LOADER_SUFFIX;
#endif

#ifdef _WIN32
        // Get required size for wide string
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, libpath.c_str(), -1, nullptr, 0);
        // Create a buffer to hold wide characters
        std::wstring wide(size_needed, 0);
        // Convert to wide string
        MultiByteToWideChar(CP_UTF8, 0, libpath.c_str(), -1, &wide[0], size_needed);
        // Get const wchar_t*
        const wchar_t* path = wide.c_str();

        DWORD handle = 0;
        DWORD size = GetFileVersionInfoSizeW(path, &handle);
        std::vector<BYTE> data(size);
        if (GetFileVersionInfoW(path, handle, size, data.data())) {
            VS_FIXEDFILEINFO* loader_version = NULL;
            uint32_t loader_version_size = 0;
            if (VerQueryValueW(data.data(), L"\\", (LPVOID*)&loader_version, &loader_version_size) ||
                !loader_version_size) {
                // Version is in dwFileVersionMS (high: major.minor) and dwFileVersionLS (low: build.revision)
                version = loader_version->dwFileVersionMS;
            }
        }
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

const uint32_t ZeroApi::getVersion() {
    return version;
}

}  // namespace intel_npu
