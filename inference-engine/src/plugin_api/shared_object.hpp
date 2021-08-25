// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 * @file ie_system_conf.h
 */

#pragma once

#include "ie_api.h"

namespace ov {
namespace runtime {
struct INFERENCE_ENGINE_API_CLASS(SharedObject) {
    void* shared_object = nullptr;

    /**
     * @brief Loads a library with the name specified.
     * @param path Full or relative path to the plugin library
     */
    explicit SharedObject(const char* path);

#ifdef ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Loads a library with the wide char name specified.
     * @param path Full or relative path to the plugin library
     */
    explicit SharedObject(const wchar_t* path);
#endif  // ENABLE_UNICODE_PATH_SUPPORT

    ~SharedObject();

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of the function to find
     * @return A pointer to the function if found
     * @throws Exception if the function is not found
     */
    void* get_symbol(const char* symbolName) const;
};
}  // namespace runtime
}  // namespace ov
