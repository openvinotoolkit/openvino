// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "so_loader.hpp"

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

namespace InferenceEngine {
namespace details {

typedef DWORD (*GetDllDirectoryA_Fnc)(DWORD, LPSTR);
typedef DWORD (*GetDllDirectoryW_Fnc)(DWORD, LPWSTR);

static GetDllDirectoryA_Fnc IEGetDllDirectoryA;
static GetDllDirectoryW_Fnc IEGetDllDirectoryW;

/**
 * @brief WINAPI based implementation for loading a shared object
 */
class SOLoader::Impl {
private:
    HMODULE shared_object;

    void LoadSymbols() {
        static std::once_flag loadFlag;
        std::call_once(loadFlag, [&]() {
            if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
                IEGetDllDirectoryA = reinterpret_cast<GetDllDirectoryA_Fnc>(GetProcAddress(hm, "GetDllDirectoryA"));
                IEGetDllDirectoryW = reinterpret_cast<GetDllDirectoryW_Fnc>(GetProcAddress(hm, "GetDllDirectoryW"));
            }
        });
    }

    // Exclude current directory from DLL search path process wise.
    // If application specific path was configured before then
    // current directory is already excluded.
    // GetDLLDirectory does not distinguish if application specific
    // path was set to "" or NULL so reset it to "" to keep
    // application safe.
    void ExcludeCurrentDirectoryA() {
#if !WINAPI_PARTITION_SYSTEM
        LoadSymbols();
        if (IEGetDllDirectoryA && IEGetDllDirectoryA(0, NULL) <= 1) {
            SetDllDirectoryA("");
        }
#endif
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    void ExcludeCurrentDirectoryW() {
#    if !WINAPI_PARTITION_SYSTEM
        LoadSymbols();
        if (IEGetDllDirectoryW && IEGetDllDirectoryW(0, NULL) <= 1) {
            SetDllDirectoryW(L"");
        }
#    endif
    }
#endif

    static const char kPathSeparator = '\\';

    static const char* FindLastPathSeparator(LPCSTR path) {
        const char* const last_sep = strchr(path, kPathSeparator);
        return last_sep;
    }

    static std::string GetDirname(LPCSTR path) {
        auto pos = FindLastPathSeparator(path);
        if (pos == nullptr) {
            return path;
        }
        std::string original(path);
        original[pos - path] = 0;
        return original;
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    static const wchar_t* FindLastPathSeparator(LPCWSTR path) {
        const wchar_t* const last_sep = wcsrchr(path, kPathSeparator);
        return last_sep;
    }

    static std::wstring GetDirname(LPCWSTR path) {
        auto pos = FindLastPathSeparator(path);
        if (pos == nullptr) {
            return path;
        }
        std::wstring original(path);
        original[pos - path] = 0;
        return original;
    }

    void LoadPluginFromDirectoryW(LPCWSTR path) {
#    if !WINAPI_PARTITION_SYSTEM
        LoadSymbols();
        if (IEGetDllDirectoryW) {
            DWORD nBufferLength = IEGetDllDirectoryW(0, NULL);
            std::vector<WCHAR> lpBuffer(nBufferLength);
            IEGetDllDirectoryW(nBufferLength, &lpBuffer.front());

            auto dirname = GetDirname(path);
            SetDllDirectoryW(dirname.c_str());
            shared_object = LoadLibraryW(path);

            SetDllDirectoryW(&lpBuffer.front());
        }
#    endif
    }
#endif
    void LoadPluginFromDirectoryA(LPCSTR path) {
#if !WINAPI_PARTITION_SYSTEM
        LoadSymbols();
        if (IEGetDllDirectoryA) {
            DWORD nBufferLength = IEGetDllDirectoryA(0, NULL);
            std::vector<CHAR> lpBuffer(nBufferLength);
            IEGetDllDirectoryA(nBufferLength, &lpBuffer.front());

            auto dirname = GetDirname(path);
            SetDllDirectoryA(dirname.c_str());
            shared_object = LoadLibraryA(path);

            SetDllDirectoryA(&lpBuffer.front());
        }
#endif
    }

public:
    /**
     * @brief A shared pointer to SOLoader
     */
    using Ptr = std::shared_ptr<SOLoader>;

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Loads a library with the name specified. The library is loaded according to the
     *        WinAPI LoadLibrary rules
     * @param pluginName Full or relative path to the plugin library
     */
    explicit Impl(const wchar_t* pluginName) {
        ExcludeCurrentDirectoryW();
        LoadPluginFromDirectoryW(pluginName);

        if (!shared_object) {
            shared_object = LoadLibraryW(pluginName);
        }

        if (!shared_object) {
            char cwd[1024];
            IE_THROW() << "Cannot load library '" << FileUtils::wStringtoMBCSstringChar(std::wstring(pluginName))
                       << "': " << GetLastError() << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        }
    }
#endif

    explicit Impl(const char* pluginName) {
        ExcludeCurrentDirectoryA();
        LoadPluginFromDirectoryA(pluginName);

        if (!shared_object) {
            shared_object = LoadLibraryA(pluginName);
        }

        if (!shared_object) {
            char cwd[1024];
            IE_THROW() << "Cannot load library '" << pluginName << "': " << GetLastError()
                       << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        }
    }

    ~Impl() {
        FreeLibrary(shared_object);
    }

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of function to find
     * @return A pointer to the function if found
     * @throws Exception if the function is not found
     */
    void* get_symbol(const char* symbolName) const {
        if (!shared_object) {
            IE_THROW() << "Cannot get '" << symbolName << "' content from unknown library!";
        }
        auto procAddr = reinterpret_cast<void*>(GetProcAddress(shared_object, symbolName));
        if (procAddr == nullptr)
            IE_THROW(NotFound) << "GetProcAddress cannot locate method '" << symbolName << "': " << GetLastError();

        return procAddr;
    }
};

SOLoader::~SOLoader() {}

SOLoader::SOLoader(const char* pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#ifdef ENABLE_UNICODE_PATH_SUPPORT
SOLoader::SOLoader(const wchar_t* pluginName) {
    _impl = std::make_shared<Impl>(pluginName);
}
#endif

void* SOLoader::get_symbol(const char* symbolName) const {
    if (_impl == nullptr) {
        IE_THROW(NotAllocated) << "SOLoader is not initialized";
    }
    return _impl->get_symbol(symbolName);
}

}  // namespace details
}  // namespace InferenceEngine

