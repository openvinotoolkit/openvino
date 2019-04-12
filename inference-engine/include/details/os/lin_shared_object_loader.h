// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief POSIX compatible loader for a shared object
 * @file lin_shared_object_loader.h
 */
#pragma once

#include <dlfcn.h>

#include "../../ie_api.h"
#include "../ie_exception.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief This class provides an OS shared module abstraction
 */
class SharedObjectLoader {
private:
    void *shared_object = nullptr;

public:
    /**
     * @brief Loads a library with the name specified. The library is loaded according to
     *        the POSIX rules for dlopen
     * @param pluginName Full or relative path to the library
     */
    explicit SharedObjectLoader(const char* pluginName) {
        shared_object = dlopen(pluginName, RTLD_LAZY);

        if (shared_object == nullptr)
            THROW_IE_EXCEPTION << "Cannot load library '" << pluginName << "': " << dlerror();
    }
    ~SharedObjectLoader() noexcept(false) {
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
    void *get_symbol(const char* symbolName) const {
        void * procAddr = nullptr;

        procAddr = dlsym(shared_object, symbolName);
        if (procAddr == nullptr)
            THROW_IE_EXCEPTION << "dlSym cannot locate method '" << symbolName << "': " << dlerror();
        return procAddr;
    }
};

}  // namespace details
}  // namespace InferenceEngine
