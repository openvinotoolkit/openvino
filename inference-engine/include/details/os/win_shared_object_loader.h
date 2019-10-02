// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief WINAPI compatible loader for a shared object
 * @file win_shared_object_loader.h
 */
#pragma once

#include "../../ie_api.h"
#include "../ie_exception.hpp"

// Avoidance of Windows.h to include winsock library.
#define _WINSOCKAPI_
// Avoidance of Windows.h to define min/max.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <direct.h>

namespace InferenceEngine {
namespace details {

/**
 * @brief This class provides an OS shared module abstraction
 */
class SharedObjectLoader {
private:
    HMODULE shared_object;

public:
    /**
     * @brief Loads a library with the name specified. The library is loaded according to the
     *        WinAPI LoadLibrary rules
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(LPCTSTR pluginName) {
        char cwd[1024];
        // Exclude current directory from DLL search path process wise.
        // If application specific path was configured before then
        // current directory is alread excluded.
        // GetDLLDirectory does not distinguish if aplication specific
        // path was set to "" or NULL so reset it to "" to keep
        // aplication safe.
        if (GetDllDirectory(0, NULL) <= 1) {
            SetDllDirectory(
#if defined UNICODE
                L"");
#else
                "");
#endif
        }
        shared_object = LoadLibrary(pluginName);
        if (!shared_object) {
            THROW_IE_EXCEPTION << "Cannot load library '"
                << pluginName << "': "
                << GetLastError()
                << " from cwd: " << _getcwd(cwd, 1024);
        }
    }
    ~SharedObjectLoader() {
        FreeLibrary(shared_object);
    }

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of function to find
     * @return A pointer to the function if found
     * @throws InferenceEngineException if the function is not found
     */
    void *get_symbol(const char* symbolName) const {
        if (!shared_object) {
            THROW_IE_EXCEPTION << "Cannot get '" << symbolName << "' content from unknown library!";
        }
        auto procAddr = reinterpret_cast<void*>(GetProcAddress(shared_object, symbolName));
        if (procAddr == nullptr)
            THROW_IE_EXCEPTION << "GetProcAddress cannot locate method '" << symbolName << "': " << GetLastError();

        return procAddr;
    }
};

}  // namespace details
}  // namespace InferenceEngine
