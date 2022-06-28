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

/**
 * \brief Close shared object (library) with defer.
 *
 * Can be used as custom deleter in smart pointer.
 */
class SharedObjectDeferCloser {
public:
    SharedObjectDeferCloser();  //!< Ctor

    /**
     * @brief Close shared object (library) with defer.
     *
     * Starts thread which complete closing the shared object.
     *
     * @param shared_object pointer to shared_object.
     */
    void operator()(void* shared_object) const;

private:
    static int m_count;                       //!< Number of libraries opened.
    static std::mutex m_count_mutex;          //!< Mutex to guard access to counter.
    static std::once_flag m_atexit_register;  //!< Flag to register wait for lib close at application exit.

    /**
     * @brief Close the library with delay.
     *
     * @param shared_object pointer to shared object (library).
     */
    static inline void lib_delayed_close(void* shared_object);

    /**
     * @brief Check if any library is open.
     *
     * @return True when any library open, otherwise false.
     */
    static inline bool any_open();

    /**
     * @brief Waits until all libraries are closed.
     *
     * Function waits until all threads which should close libraries will end.
     */
    static inline void wait_all_closed();
};

}  // namespace util
}  // namespace ov
