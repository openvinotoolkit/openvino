// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 * @file shared_object.hpp
 */

#pragma once

#include <memory>

#include "openvino/util/util.hpp"

namespace ov {
namespace util {

/**
 * @brief Loads a library with the name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
std::shared_ptr<void> load_shared_object(const char* path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Loads a library with the wide char name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
std::shared_ptr<void> load_shared_object(const wchar_t* path);
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Searches for a function symbol in the loaded module
 * @param shared_object shared object reference
 * @param symbolName Name of the function to find
 * @return A pointer to the function if found
 * @throws Exception if the function is not found
 */
void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName);

}  // namespace util
}  // namespace ov
