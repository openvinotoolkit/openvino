// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 *
 * @file ie_so_loader.h
 */
#pragma once

#include <memory>

#include "ie_api.h"

namespace InferenceEngine {
namespace details {

/**
 * @brief This class provides an OS shared module abstraction
 */
class INFERENCE_ENGINE_API_CLASS(SharedObjectLoader) {
    class Impl;
    std::shared_ptr<Impl> _impl;

public:
    /**
     * @brief Default constructor
     */
    SharedObjectLoader() = default;

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Loads a library with the wide char name specified.
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(const wchar_t* pluginName);
#endif

    /**
     * @brief Loads a library with the name specified.
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(const char * pluginName);

    /**
     * @brief A destructor
     */
    ~SharedObjectLoader();

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of function to find
     * @return A pointer to the function if found
     * @throws Exception if the function is not found
     */
    void* get_symbol(const char* symbolName) const;
};

}  // namespace details
}  // namespace InferenceEngine
