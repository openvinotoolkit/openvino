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

class SharedObjectLoader::Impl {
private:
    HMODULE shared_object;

    void ExcludeCurrentDirectoryA() {
        // Exclude current directory from DLL search path process wise.
        // If application specific path was configured before then
        // current directory is alread excluded.
        // GetDLLDirectory does not distinguish if aplication specific
        // path was set to "" or NULL so reset it to "" to keep
        // aplication safe.
        if (GetDllDirectoryA(0, NULL) <= 1) {
            SetDllDirectoryA("");
        }
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    void ExcludeCurrentDirectoryW() {
        // Exclude current directory from DLL search path process wise.
        // If application specific path was configured before then
        // current directory is alread excluded.
        // GetDLLDirectory does not distinguish if aplication specific
        // path was set to "" or NULL so reset it to "" to keep
        // aplication safe.
        if (GetDllDirectoryW(0, NULL) <= 1) {
            SetDllDirectoryW(TEXT(""));
        }
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
