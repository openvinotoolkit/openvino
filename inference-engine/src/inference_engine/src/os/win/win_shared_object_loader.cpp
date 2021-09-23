// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_common.h"
#include "details/ie_so_loader.h"
#include "file_utils.h"
#include "shared_object.hpp"

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
# error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#endif

#include <mutex>
#include <direct.h>

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <windows.h>

namespace ov {
namespace runtime {
std::shared_ptr<void> load_shared_object(const char* path) {
    void* shared_object = nullptr;
    using GetDllDirectoryA_Fnc = DWORD(*)(DWORD, LPSTR);
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
        auto dirname = [path] {
            auto pos = strchr(path, '\\');
            if (pos == nullptr) {
                return std::string{path};
            }
            std::string original(path);
            original[pos - path] = 0;
            return original;
        } ();

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
        IE_THROW() << "Cannot load library '" << path << "': " << GetLastError()
            << " from cwd: " << _getcwd(cwd, sizeof(cwd));
    }
    return {shared_object,
            [] (void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<void> load_shared_object(const wchar_t* path) {
    void* shared_object = nullptr;
    using GetDllDirectoryW_Fnc = DWORD(*)(DWORD, LPWSTR);
    static GetDllDirectoryW_Fnc IEGetDllDirectoryW = nullptr;
    if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
        IEGetDllDirectoryW = reinterpret_cast<GetDllDirectoryW_Fnc>(GetProcAddress(hm, "GetDllDirectoryW"));
    }
    // ExcludeCurrentDirectory
#if !WINAPI_PARTITION_SYSTEM
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
        } ();
        SetDllDirectoryW(dirname.c_str());
        shared_object = LoadLibraryW(path);

        SetDllDirectoryW(&lpBuffer.front());
    }
#endif
    if (!shared_object) {
        shared_object = LoadLibraryW(path);
    }
    if (!shared_object) {
        char cwd[1024];
        IE_THROW() << "Cannot load library '" << ov::util::wstring_to_string(std::wstring(path)) << "': " << GetLastError()
                            << " from cwd: " << _getcwd(cwd, sizeof(cwd));
    }
    return {shared_object,
            [] (void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
}
#endif

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name) {
    if (!shared_object) {
        IE_THROW() << "Cannot get '" << symbol_name << "' content from unknown library!";
    }
    auto procAddr = reinterpret_cast<void*>(GetProcAddress(
        reinterpret_cast<HMODULE>(const_cast<void*>(shared_object.get())), symbol_name));
    if (procAddr == nullptr) {
        IE_THROW(NotFound)
            << "GetProcAddress cannot locate method '" << symbol_name << "': " << GetLastError();
    }
    return procAddr;
}
}  // namespace runtime
}  // namespace ov

namespace InferenceEngine {
namespace details {
struct SharedObjectLoader::Impl {
    std::shared_ptr<void> shared_object = nullptr;

    explicit Impl(const std::shared_ptr<void>& shared_object_) : shared_object{shared_object_} {}

    explicit Impl(const char* pluginName) : shared_object{ov::runtime::load_shared_object(pluginName)} {}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    explicit Impl(const wchar_t* pluginName) : shared_object{ov::runtime::load_shared_object(pluginName)} {}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

    void* get_symbol(const char* symbolName) const {
        return ov::runtime::get_symbol(shared_object, symbolName);
    }
};

SharedObjectLoader::SharedObjectLoader(const std::shared_ptr<void>& shared_object) {
    _impl.reset(new Impl(shared_object));
}

SharedObjectLoader::~SharedObjectLoader() {}

SharedObjectLoader::SharedObjectLoader(const char * pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
SharedObjectLoader::SharedObjectLoader(const wchar_t* pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#endif

void* SharedObjectLoader::get_symbol(const char* symbolName) const {
    if (_impl == nullptr) {
        IE_THROW(NotAllocated) << "SharedObjectLoader is not initialized";
    }
    return _impl->get_symbol(symbolName);
}

std::shared_ptr<void> SharedObjectLoader::get() const {
    return _impl->shared_object;
}

}  // namespace details
}  // namespace InferenceEngine
