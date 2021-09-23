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
/**
 * @brief Loads a library with the name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
INFERENCE_ENGINE_API_CPP(std::shared_ptr<void>) load_shared_object(const char* path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Loads a library with the wide char name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
INFERENCE_ENGINE_API_CPP(std::shared_ptr<void>) load_shared_object(const wchar_t* path);
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Searches for a function symbol in the loaded module
 * @param shared_object shared object reference
 * @param symbolName Name of the function to find
 * @return A pointer to the function if found
 * @throws Exception if the function is not found
 */
INFERENCE_ENGINE_API_CPP(void*) get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName);
}  // namespace runtime
}  // namespace ov
