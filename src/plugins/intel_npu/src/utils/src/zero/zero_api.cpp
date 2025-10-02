// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_api.hpp"

#ifdef _WIN32
#    include <windows.h>
#    include <winver.h>
#endif

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {
ZeroApi::ZeroApi() {
    const std::string base_name = "ze_loader";
    try {
        auto lib_path = ov::util::make_plugin_library_name({}, base_name);
#if !defined(_WIN32) && !defined(ANDROID)
        lib_path = lib_path + LIB_ZE_LOADER_SUFFIX;
#endif

        this->lib = ov::util::load_shared_object(lib_path.c_str());

#ifdef _WIN32
        DWORD handle = 0;
#    if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
        const wchar_t* wide_path = ov::util::string_to_wstring(lib_path).c_str();
        DWORD size = GetFileVersionInfoSizeW(wide_path, &handle);
        if (size > 0) {
            std::vector<BYTE> data(size);
            if (GetFileVersionInfoW(wide_path, handle, size, data.data())) {
                VS_FIXEDFILEINFO* loader_version = NULL;
                uint32_t loader_version_size = 0;
                if (VerQueryValueW(data.data(), L"\\", (LPVOID*)&loader_version, &loader_version_size) &&
                    loader_version_size > 0 && loader_version != nullptr) {
                    // Version is in dwFileVersionMS (high: major.minor) and dwFileVersionLS (low: build.revision)
                    version = loader_version->dwFileVersionMS;
                }
            }
        }
#    else
        DWORD size = GetFileVersionInfoSizeA(lib_path.c_str(), &handle);
        if (size > 0) {
            std::vector<BYTE> data(size);
            if (GetFileVersionInfoA(lib_path.c_str(), handle, size, data.data())) {
                VS_FIXEDFILEINFO* loader_version = NULL;
                uint32_t loader_version_size = 0;
                if (VerQueryValueA(data.data(), "\\", (LPVOID*)&loader_version, &loader_version_size) &&
                    loader_version_size > 0 && loader_version != nullptr) {
                    // Version is in dwFileVersionMS (high: major.minor) and dwFileVersionLS (low: build.revision)
                    version = loader_version->dwFileVersionMS;
                }
            }
        }
#    endif
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
