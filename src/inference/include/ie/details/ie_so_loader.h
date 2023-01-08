// Copyright (C) 2018-2022 Intel Corporation
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
 * @deprecated This is internal stuff. Use Inference Engine Plugin API
 * @brief This class provides an OS shared module abstraction
 */
class INFERENCE_ENGINE_DEPRECATED("This is internal stuff. Use Inference Engine Plugin API")
    INFERENCE_ENGINE_API_CLASS(SharedObjectLoader) {
    std::shared_ptr<void> _so;

public:
    /**
     * @brief Constructs from existing object
     */
    SharedObjectLoader(const std::shared_ptr<void>& so);

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
    explicit SharedObjectLoader(const char* pluginName);

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

    /**
     * @brief Retruns reference to type erased implementation
     * @throws Exception if the function is not found
     */
    std::shared_ptr<void> get() const;
};

}  // namespace details
}  // namespace InferenceEngine
