// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>

#include "details/ie_exception.hpp"
#include "details/ie_so_loader.h"
#include "file_utils.h"

namespace InferenceEngine {
namespace details {

class SharedObjectLoader::Impl {
private:
    void* shared_object = nullptr;

public:
    explicit Impl(const char* pluginName) {
        shared_object = dlopen(pluginName, RTLD_LAZY);

        if (shared_object == nullptr)
            THROW_IE_EXCEPTION << "Cannot load library '" << pluginName << "': " << dlerror();
    }

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    explicit Impl(const wchar_t* pluginName) : Impl(FileUtils::wStringtoMBCSstringChar(pluginName).c_str()) {
    }
#endif  // ENABLE_UNICODE_PATH_SUPPORT

    ~Impl() noexcept(false) {
        if (0 != dlclose(shared_object)) {
            THROW_IE_EXCEPTION << "dlclose failed: " << dlerror();
        }
    }

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of the function to find
     * @return A pointer to the function if found
     * @throws InferenceEngineException if the function is not found
     */
    void* get_symbol(const char* symbolName) const {
        void* procAddr = nullptr;

        procAddr = dlsym(shared_object, symbolName);
        if (procAddr == nullptr)
            THROW_IE_EXCEPTION << "dlSym cannot locate method '" << symbolName << "': " << dlerror();
        return procAddr;
    }
};

#ifdef ENABLE_UNICODE_PATH_SUPPORT
SharedObjectLoader::SharedObjectLoader(const wchar_t* pluginName) {
    _impl.reset(new Impl(pluginName));
}
#endif

SharedObjectLoader::SharedObjectLoader(const char * pluginName) {
    _impl.reset(new Impl(pluginName));
}

SharedObjectLoader::~SharedObjectLoader() noexcept(false) {
}

void* SharedObjectLoader::get_symbol(const char* symbolName) const {
    return _impl->get_symbol(symbolName);
}

}  // namespace details
}  // namespace InferenceEngine
