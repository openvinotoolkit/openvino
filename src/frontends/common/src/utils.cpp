// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/util/file_util.hpp"
#include "plugin_loader.hpp"

#ifndef _WIN32
#    include <dlfcn.h>
#    include <limits.h>
#    include <unistd.h>
#else
#    if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#        error "Only WINAPI_PARTITION_DESKTOP is supported, because of GetModuleHandleEx[A|W]"
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace {

static std::string _get_frontend_library_path() {
#ifdef _WIN32
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    WCHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(ov::frontend::get_frontend_library_path),
                            &hm)) {
        FRONT_END_INITIALIZATION_CHECK(false, "GetModuleHandle returned ", GetLastError());
    }
    GetModuleFileNameW(hm, (LPWSTR)ie_library_path, sizeof(ie_library_path) / sizeof(ie_library_path[0]));
    return ov::util::wstring_to_string(ov::util::get_directory(std::wstring(ie_library_path)));
#    else
    CHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPSTR>(ov::frontend::get_frontend_library_path),
                            &hm)) {
        FRONT_END_INITIALIZATION_CHECK(false, "GetModuleHandle returned ", GetLastError());
    }
    GetModuleFileNameA(hm, (LPSTR)ie_library_path, sizeof(ie_library_path));
    return ov::util::get_directory(std::string(ie_library_path));
#    endif
#elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(ov::frontend::get_frontend_library_path), &info);
    return ov::util::get_directory(ov::util::get_absolute_file_path(std::string(info.dli_fname))).c_str();
#else
#    error "Unsupported OS"
#endif  // _WIN32
}
}  // namespace

std::string ov::frontend::get_frontend_library_path() {
    return _get_frontend_library_path();
}
