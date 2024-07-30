// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

//
// LoadLibraryA, LoadLibraryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleExA, GetModuleHandleExW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleA, GetModuleHandleW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// SetDllDirectoryA, SetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// GetDllDirectoryA, GetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// SetupDiGetClassDevsA, SetupDiEnumDeviceInfo, SetupDiGetDeviceInstanceIdA, SetupDiDestroyDeviceInfoList:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL (default)
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//

#if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#    error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#endif

#include <direct.h>

#include <mutex>

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>

namespace ov {
namespace util {
std::shared_ptr<void> load_shared_object(const char* path) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(path).c_str());
#else
    void* shared_object = nullptr;
    using GetDllDirectoryA_Fnc = DWORD (*)(DWORD, LPSTR);
    GetDllDirectoryA_Fnc IEGetDllDirectoryA = nullptr;
    if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
        IEGetDllDirectoryA = reinterpret_cast<GetDllDirectoryA_Fnc>(GetProcAddress(hm, "GetDllDirectoryA"));
    }
#if !WINAPI_PARTITION_SYSTEM
    // ExcludeCurrentDirectory
    if (IEGetDllDirectoryA && IEGetDllDirectoryA(0, NULL) <= 1) {
        SetDllDirectoryA("");
    }
    // LoadPluginFromDirectory
    if (IEGetDllDirectoryA) {
        DWORD nBufferLength = IEGetDllDirectoryA(0, NULL);
        std::vector<CHAR> lpBuffer(nBufferLength);
        IEGetDllDirectoryA(nBufferLength, &lpBuffer.front());

        // GetDirname
        auto dirname = get_directory(path);

        SetDllDirectoryA(dirname.c_str());
        shared_object = LoadLibraryA(path);

        SetDllDirectoryA(&lpBuffer.front());
    }
#endif
    if (!shared_object) {
        shared_object = LoadLibraryA(path);
    }

    if (!shared_object) {
        char cwd[1024];
        std::stringstream ss;
        ss << "Cannot load library '" << path << "': " << GetLastError() << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        throw std::runtime_error(ss.str());
    }
    return {shared_object, [](void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
#endif
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<void> load_shared_object(const wchar_t* path) {
    void* shared_object = nullptr;
    using GetDllDirectoryW_Fnc = DWORD (*)(DWORD, LPWSTR);
    static GetDllDirectoryW_Fnc IEGetDllDirectoryW = nullptr;
    if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
        IEGetDllDirectoryW = reinterpret_cast<GetDllDirectoryW_Fnc>(GetProcAddress(hm, "GetDllDirectoryW"));
    }
    // ExcludeCurrentDirectory
#    if !WINAPI_PARTITION_SYSTEM
    if (IEGetDllDirectoryW && IEGetDllDirectoryW(0, NULL) <= 1) {
        SetDllDirectoryW(L"");
    }
    if (IEGetDllDirectoryW) {
        DWORD nBufferLength = IEGetDllDirectoryW(0, NULL);
        std::vector<WCHAR> lpBuffer(nBufferLength);
        IEGetDllDirectoryW(nBufferLength, &lpBuffer.front());
        auto dirname = [path] {
            auto pos = wcsrchr(path, '\\');
            if (pos == nullptr) {
                return std::wstring{path};
            }
            std::wstring original(path);
            original[pos - path] = 0;
            return original;
        }();
        SetDllDirectoryW(dirname.c_str());
        shared_object = LoadLibraryW(path);

        SetDllDirectoryW(&lpBuffer.front());
    }
#    endif
    if (!shared_object) {
        shared_object = LoadLibraryW(path);
    }
    if (!shared_object) {
        char cwd[1024];
        std::stringstream ss;
        ss << "Cannot load library '" << ov::util::wstring_to_string(std::wstring(path)) << "': " << GetLastError()
           << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        throw std::runtime_error(ss.str());
    }
    return {shared_object, [](void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
}
#endif

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name) {
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot get '" << symbol_name << "' content from unknown library!";
        throw std::runtime_error(ss.str());
    }
    auto procAddr = reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(const_cast<void*>(shared_object.get())), symbol_name));
    if (procAddr == nullptr) {
        std::stringstream ss;
        ss << "GetProcAddress cannot locate method '" << symbol_name << "': " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    return procAddr;
}
}  // namespace util
}  // namespace ov
