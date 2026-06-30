// Copyright (C) 2018-2026 Intel Corporation
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
#include <psapi.h> 

namespace ov::util {

std::shared_ptr<void> load_shared_object(const std::filesystem::path& path) {
        // ── memory snapshot BEFORE ──────────────────────────────
    PROCESS_MEMORY_COUNTERS pmc_before{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc_before, sizeof(pmc_before));
    // ────────────────────────────────────────────────────────
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
        const auto& dir_name = path.has_parent_path() ? path.parent_path() : path;
        SetDllDirectoryW(dir_name.c_str());
        shared_object = LoadLibraryW(path.c_str());

        SetDllDirectoryW(&lpBuffer.front());
    }
#    endif
    PROCESS_MEMORY_COUNTERS pmc_after{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc_after, sizeof(pmc_after));
    if (!shared_object) {
        shared_object = LoadLibraryW(path.c_str());
    }
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot load library \"" << path_to_string(path) << "\": " << GetLastError()
           << " from cwd: " << path_to_string(std::filesystem::current_path());
        throw std::runtime_error(ss.str());
    }

    SIZE_T delta = (pmc_after.WorkingSetSize > pmc_before.WorkingSetSize)
                    ? (pmc_after.WorkingSetSize - pmc_before.WorkingSetSize)
                    : 0;

    printf("[MemTrack] %s\n"
           "  WorkingSet before : %6zu KB\n"
           "  WorkingSet after  : %6zu KB\n"
           "  Delta             : %6zu KB\n"
           "  PrivateBytes after: %6zu KB\n",
           path.filename().string().c_str(),
           pmc_before.WorkingSetSize  ,
           pmc_after.WorkingSetSize   ,
           delta                      ,
           pmc_after.PagefileUsage    );
    // ── memory snapshot AFTER ───────────────────────────────
    // ────────────────────────────────────────────────────────
    return {shared_object, [](void* shared_object) {
        PROCESS_MEMORY_COUNTERS pmc_before_unload{};
        GetProcessMemoryInfo(GetCurrentProcess(), 
                             &pmc_before_unload, 
                             sizeof(pmc_before_unload));

        FreeLibrary(reinterpret_cast<HMODULE>(shared_object));

        PROCESS_MEMORY_COUNTERS pmc_after_unload{};
        GetProcessMemoryInfo(GetCurrentProcess(), 
                             &pmc_after_unload, 
                             sizeof(pmc_after_unload));

        SIZE_T freed = (pmc_before_unload.WorkingSetSize > pmc_after_unload.WorkingSetSize)
                        ? (pmc_before_unload.WorkingSetSize - pmc_after_unload.WorkingSetSize)
                        : 0;

        printf("[MemTrack] FreeLibrary\n"
               "  WorkingSet before : %6zu KB\n"
               "  WorkingSet after  : %6zu KB\n"
               "  Freed             : %6zu KB\n",
               pmc_before_unload.WorkingSetSize / 1024,
               pmc_after_unload.WorkingSetSize  / 1024,
               freed                            / 1024);
        // ──────────────────────────────────────────────────
            }};
}


void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name) {
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot get '" << symbol_name << "' content from unknown library!";
        throw std::runtime_error(ss.str());
    }
    auto proc_addr = reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(const_cast<void*>(shared_object.get())), symbol_name));
    if (proc_addr == nullptr) {
        std::stringstream ss;
        ss << "GetProcAddress cannot locate method '" << symbol_name << "': " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    return proc_addr;
}
}  // namespace ov::util
