// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include "frontend_manager/frontend_exceptions.hpp"
#include "plugin_loader.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <codecvt>
#include <locale>
#endif
#else
#if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#error "Only WINAPI_PARTITION_DESKTOP is supported, because of GetModuleHandleEx[A|W]"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace
{
    std::string getPathName(const std::string& s)
    {
        size_t i = s.rfind(FileSeparator, s.length());
        if (i != std::string::npos)
        {
            return (s.substr(0, i));
        }

        return {};
    }

} // namespace

static std::string _getFrontendLibraryPath()
{
#ifdef _WIN32
    CHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPSTR>(ngraph::frontend::getFrontendLibraryPath),
                            &hm))
    {
        FRONT_END_INITIALIZATION_CHECK(false, "GetModuleHandle returned ", GetLastError());
    }
    GetModuleFileNameA(hm, (LPSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::string(ie_library_path));
#elif defined(__APPLE__) || defined(__linux__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(ngraph::frontend::getFrontendLibraryPath), &info);
    return getPathName(std::string(info.dli_fname)).c_str();
#else
#error "Unsupported OS"
#endif // _WIN32
}

std::string ngraph::frontend::getFrontendLibraryPath()
{
    return _getFrontendLibraryPath();
}
