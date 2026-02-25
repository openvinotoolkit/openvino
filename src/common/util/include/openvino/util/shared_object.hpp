// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 * @file shared_object.hpp
 */

#pragma once

#include <filesystem>
#include <memory>

namespace ov::util {

/**
 * @brief Loads a library with the name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
std::shared_ptr<void> load_shared_object(const std::filesystem::path& path);

/**
 * @brief Searches for a function symbol in the loaded module
 * @param shared_object shared object reference
 * @param symbol_name Name of the function to find
 * @return A pointer to the function if found
 * @throws Exception if the function is not found
 */
void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name);
}  // namespace ov::util
