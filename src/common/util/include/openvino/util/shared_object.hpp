// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 * @file shared_object.hpp
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "openvino/util/util.hpp"

namespace ov {
namespace util {
/**
 * \brief Close shared object (library).
 *
 * Can be used as custom deleter in smart pointer.
 */
class SharedObjectCloser {
public:
    /**
     * @brief Closes shared object at given pointer.
     *
     * @param shared_object pointer to shared object.
     */
    void operator()(void* shared_object) const;
};

/**
 * @brief Loads a library with the name specified.
 *
 * @param path             Full or relative path to the plugin library
 * @param sh_object_closer custom closer of shared_object.
 *
 * @return Reference to shared object
 */
std::shared_ptr<void> load_shared_object(const char* path,
                                         std::function<void(void*)> sh_object_closer = SharedObjectCloser());

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Loads a library with the wide char name specified.
 *
 * @param path             Full or relative path to the plugin library
 * @param sh_object_closer Custom closer of shared_object.
 *
 * @return Reference to shared object
 */
std::shared_ptr<void> load_shared_object(const wchar_t* path,
                                         std::function<void(void*)> sh_object_closer = SharedObjectCloser());
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Searches for a function symbol in the loaded module
 *
 * @param shared_object shared object reference
 * @param symbolName    Name of the function to find
 *
 * @return A pointer to the function if found
 * @throws Exception if the function is not found
 */
void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName);
}  // namespace util
}  // namespace ov
