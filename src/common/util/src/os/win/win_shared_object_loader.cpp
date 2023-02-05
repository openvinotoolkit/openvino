// Copyright (C) 2018-2023 Intel Corporation
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
#include <wintrust.h>
#include <Softpub.h>

// #include <wincrypt.h>
// #include <wintrust.h>

// #pragma comment (lib, "wintrust")

namespace {
bool verify_embedded_signature(LPCWSTR pwszSourceFile)
{
	GUID WintrustVerifyGuid = WINTRUST_ACTION_GENERIC_VERIFY_V2;

	WINTRUST_DATA wd = { 0 };
	WINTRUST_FILE_INFO wfi = { 0 };

	memset(&wfi, 0, sizeof(wfi));
	wfi.cbStruct = sizeof(WINTRUST_FILE_INFO);
	wfi.pcwszFilePath = pwszSourceFile;
	wfi.hFile = NULL;
	wfi.pgKnownSubject = NULL;

	memset(&wd, 0, sizeof(wd));
	wd.cbStruct = sizeof(WINTRUST_DATA);
	wd.dwUnionChoice = WTD_CHOICE_FILE;
	wd.pFile = &wfi;
	wd.dwUIChoice = WTD_UI_NONE;
	wd.fdwRevocationChecks = WTD_REVOKE_NONE;
	wd.dwStateAction = WTD_STATEACTION_VERIFY;
	wd.hWVTStateData = NULL;
	wd.pwszURLReference = NULL;
	wd.pPolicyCallbackData = NULL;
	wd.pSIPClientData = NULL;
	wd.dwUIContext = 0;

    auto status = WinVerifyTrust(NULL, &WintrustVerifyGuid, &wd);

    // Any hWVTStateData must be released by a call with close.
    wd.dwStateAction = WTD_STATEACTION_CLOSE;
    std::ignore = WinVerifyTrust(NULL, &WintrustVerifyGuid, &wd);

	return status == ERROR_SUCCESS;
}

bool verify_embedded_signature(const char* path) {
    return verify_embedded_signature(ov::util::string_to_wstring(path).c_str());
}
}  // namespace

namespace ov {
namespace util {
std::shared_ptr<void> load_shared_object(const char* path, const bool& verify_signature) {
    if (verify_signature) {
        if (!is_absolute_file_path(path)) {
            // TODO: check how it works with file names
            std::stringstream ss;
            ss << "Cannot verify signature of library '" << path << "': path isn't absolute.";
            throw std::runtime_error(ss.str());
        }
        if (!verify_embedded_signature(path)) {
            std::stringstream ss;
            ss << "Signature verification of library '" << path << "' failed";
            throw std::runtime_error(ss.str());
        }
    }
    return load_shared_object(path);
}

std::shared_ptr<void> load_shared_object(const char* path) {
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
        auto dirname = [path] {
            auto pos = strchr(path, '\\');
            if (pos == nullptr) {
                return std::string{path};
            }
            std::string original(path);
            original[pos - path] = 0;
            return original;
        }();

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
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<void> load_shared_object(const wchar_t* path, const bool& verify_signature) {
    if (verify_signature) {
        if (!is_absolute_file_path(ov::util::wstring_to_string(path))) {
            // TODO: check how it works with file names
            std::stringstream ss;
            ss << "Cannot verify signature of library '" << ov::util::wstring_to_string(std::wstring(path)) << "': path isn't absolute.";
            throw std::runtime_error(ss.str());
        }
        if (!verify_embedded_signature(path)) {
            std::stringstream ss;
            ss << "Signature verification of library '" << ov::util::wstring_to_string(std::wstring(path)) << "' failed";
            throw std::runtime_error(ss.str());
        }
    }
    return load_shared_object(path);
}

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
