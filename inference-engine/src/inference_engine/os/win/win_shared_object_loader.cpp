// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_exception.hpp"
#include "details/ie_so_loader.h"
#include "file_utils.h"

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
#include <windows.h>

namespace InferenceEngine {
namespace details {

class SharedObjectLoader::Impl {
private:
    HMODULE shared_object;

    typedef DWORD(* GetDllDirectoryA_Fnc)(DWORD, LPSTR);
    typedef DWORD(* GetDllDirectoryW_Fnc)(DWORD, LPWSTR);

    static GetDllDirectoryA_Fnc IEGetDllDirectoryA;
    static GetDllDirectoryW_Fnc IEGetDllDirectoryW;

    void LoadSymbols() {
        static std::once_flag loadFlag;
        std::call_once(loadFlag, [&] () {
            if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
                IEGetDllDirectoryA = reinterpret_cast<GetDllDirectoryA_Fnc>(GetProcAddress(hm, "GetDllDirectoryA"));
                IEGetDllDirectoryW = reinterpret_cast<GetDllDirectoryW_Fnc>(GetProcAddress(hm, "GetDllDirectoryW"));
            }
        });
    }

    // Exclude current directory from DLL search path process wise.
    // If application specific path was configured before then
    // current directory is already excluded.
    // GetDLLDirectory does not distinguish if aplication specific
    // path was set to "" or NULL so reset it to "" to keep
    // application safe.
    void ExcludeCurrentDirectoryA() {
        GetDllDirectoryA;
#ifndef WINAPI_FAMILY
        LoadSymbols();
        if (IEGetDllDirectoryA && IEGetDllDirectoryA(0, NULL) <= 1) {
            SetDllDirectoryA("");
        }
#endif
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    void ExcludeCurrentDirectoryW() {
#ifndef WINAPI_FAMILY
        LoadSymbols();
        if (IEGetDllDirectoryW && IEGetDllDirectoryW(0, NULL) <= 1) {
            SetDllDirectoryW(L"");
        }
#endif
    }
#endif

public:
#ifdef ENABLE_UNICODE_PATH_SUPPORT
    explicit Impl(const wchar_t* pluginName) {
        ExcludeCurrentDirectoryW();

        shared_object = LoadLibraryW(pluginName);
        if (!shared_object) {
            char cwd[1024];
            THROW_IE_EXCEPTION << "Cannot load library '" << FileUtils::wStringtoMBCSstringChar(std::wstring(pluginName)) << "': " << GetLastError()
                               << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        }
    }
#endif

    explicit Impl(const char* pluginName) {
        ExcludeCurrentDirectoryA();

        shared_object = LoadLibraryA(pluginName);
        if (!shared_object) {
            char cwd[1024];
            THROW_IE_EXCEPTION << "Cannot load library '" << pluginName << "': " << GetLastError()
                << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        }
    }

    ~Impl() {
        FreeLibrary(shared_object);
    }

    void* get_symbol(const char* symbolName) const {
        if (!shared_object) {
            THROW_IE_EXCEPTION << "Cannot get '" << symbolName << "' content from unknown library!";
        }
        auto procAddr = reinterpret_cast<void*>(GetProcAddress(shared_object, symbolName));
        if (procAddr == nullptr)
            THROW_IE_EXCEPTION << "GetProcAddress cannot locate method '" << symbolName << "': " << GetLastError();

        return procAddr;
    }
};

#ifdef ENABLE_UNICODE_PATH_SUPPORT
SharedObjectLoader::SharedObjectLoader(const wchar_t* pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#endif

SharedObjectLoader::~SharedObjectLoader() noexcept(false) {
}

SharedObjectLoader::SharedObjectLoader(const char * pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}

void* SharedObjectLoader::get_symbol(const char* symbolName) const {
    return _impl->get_symbol(symbolName);
}

}  // namespace details
}  // namespace InferenceEngine
