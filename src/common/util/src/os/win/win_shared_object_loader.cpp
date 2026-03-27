// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

//
// LoadLibraryExW:
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

#if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#    error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibraryEx[A|W]"
#endif

#include <direct.h>

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>

// Verify that LOAD_LIBRARY_SEARCH_* flags are available.
// These require _WIN32_WINNT >= 0x0602 (Windows 8) in the Windows SDK headers.
// OpenVINO minimum supported platform is Windows 10, so this should always hold.
#if !defined(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR) || !defined(LOAD_LIBRARY_SEARCH_SYSTEM32) || \
    !defined(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
#    error \
        "LOAD_LIBRARY_SEARCH_* flags not available. Ensure _WIN32_WINNT >= 0x0602 (Windows 8) or use a Windows 10+ SDK."
#endif

namespace ov::util {

std::shared_ptr<void> load_shared_object(const std::filesystem::path& path) {
    void* shared_object = nullptr;

    // SDL436: Use LoadLibraryExW with restricted search flags to mitigate DLL injection.
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR — searches the directory containing the DLL (requires absolute path).
    // LOAD_LIBRARY_SEARCH_SYSTEM32 — allows loading system dependencies from System32.
    // This approach is thread-safe, unlike the previous SetDllDirectoryW-based method which modified
    // process-wide state and was susceptible to race conditions.
    if (path.is_absolute()) {
        shared_object = LoadLibraryExW(
            path.c_str(), NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    } else if (path.has_parent_path()) {
        auto abs_path = std::filesystem::absolute(path);
        shared_object = LoadLibraryExW(
            abs_path.c_str(), NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }

    if (!shared_object) {
        // Fallback for bare filenames or if the previous attempt failed.
        // LOAD_LIBRARY_SEARCH_DEFAULT_DIRS searches: application directory, System32, and user-added directories.
        // This avoids the insecure default search order (which includes CWD).
        shared_object = LoadLibraryExW(path.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }

    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot load library \"" << path_to_string(path) << "\": " << GetLastError()
           << " from cwd: " << path_to_string(std::filesystem::current_path());
        throw std::runtime_error(ss.str());
    }
    return {shared_object, [](void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
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
