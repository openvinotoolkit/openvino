// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_exception.hpp"
#include "details/ie_so_loader.h"
#include "file_utils.h"

#include <direct.h>
#include <windows.h>

namespace InferenceEngine {
namespace details {

/**
 * @brief WINAPI based implementation for loading a shared object
 */
class SharedObjectLoader::Impl {
 private:
    HMODULE shared_object;

    void ExcludeCurrentDirectory() {
        // Exclude current directory from DLL search path process wise.
        // If application specific path was configured before then
        // current directory is alread excluded.
        // GetDLLDirectory does not distinguish if aplication specific
        // path was set to "" or NULL so reset it to "" to keep
        // aplication safe.
        if (GetDllDirectory(0, NULL) <= 1) {
            SetDllDirectory(TEXT(""));
        }
    }

    static const char  kPathSeparator = '\\';

    static const char* FindLastPathSeparator(LPCSTR path) {
        const char* const last_sep = strchr(path, kPathSeparator);
        return last_sep;
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    static const wchar_t* FindLastPathSeparator(LPCWSTR path) {
        const char* const last_sep = wcsrchr(path, kPathSeparator);
        return last_sep;
    }

    std::basic_string<WCHAR> GetDirname(LPCWSTR path) {
        auto pos = FindLastPathSeparator(path);
        if (pos==nullptr) {
            return path;
        }
        std::basic_string<WCHAR> original(path);
        original[pos] = 0;
        return original;
    }

    std::basic_string<WCHAR> IncludePluginDirectory(LPCWSTR path) {
        std::basic_string<WCHAR> lpBuffer(path);
        DWORD nBufferLength;

        nBufferLength = GetDllDirectoryW(0, nullptr);
        std::vector<WCHAR> lpBuffer(nBufferLength);
        GetDllDirectoryW(nBufferLength, &lpBuffer.front());

        auto dirname = GetDirname(path);
        SetDllDirectoryW(dirname.c_str());

        return &lpBuffer.front();
    }
#endif
    std::basic_string<CHAR> IncludePluginDirectory(LPCSTR path) {
        std::basic_string<CHAR> lpBuffer(path);
        DWORD nBufferLength;

        nBufferLength = GetDllDirectoryW(0, nullptr);
        std::vector<CHAR> lpBuffer(nBufferLength);
        GetDllDirectoryW(nBufferLength, &lpBuffer.front());

        auto dirname = GetDirname(path);
        SetDllDirectoryW(dirname.c_str());

        return &lpBuffer.front();
    }

 public:
    /**
     * @brief A shared pointer to SharedObjectLoader
     */
    using Ptr = std::shared_ptr<SharedObjectLoader>;

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Loads a library with the name specified. The library is loaded according to the
     *        WinAPI LoadLibrary rules
     * @param pluginName Full or relative path to the plugin library
     */
    explicit Impl(const wchar_t* pluginName) {
        ExcludeCurrentDirectory();
        auto oldDir = IncludePluginDirectory(pluginName);

        shared_object = LoadLibraryW(pluginName);

        SetDllDirectoryW(oldDir.c_str());
        if (!shared_object) {
            char cwd[1024];
            THROW_IE_EXCEPTION << "Cannot load library '" << FileUtils::wStringtoMBCSstringChar(std::wstring(pluginName)) << "': " << GetLastError()
                               << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        }
    }
#endif

    explicit Impl(const char* pluginName) {
        ExcludeCurrentDirectory();
        auto oldDir = IncludePluginDirectory(pluginName);
        IncludePluginDirectory(pluginName);

        shared_object = LoadLibraryA(pluginName);

        SetDllDirectoryW(oldDir.c_str());
        if (!shared_object) {
            char cwd[1024];
            THROW_IE_EXCEPTION << "Cannot load library '" << pluginName << "': " << GetLastError()
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
     * @throws InferenceEngineException if the function is not found
     */
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
