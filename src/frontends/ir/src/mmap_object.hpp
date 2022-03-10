// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared memory map objects
 * @file mmap_object.hpp
 */

#pragma once

#include <memory>

#include "ngraph/runtime/aligned_buffer.hpp"

namespace ov {

/**
 * @brief Loads a library with the name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
std::shared_ptr<ngraph::runtime::AlignedBuffer> load_mmap_object(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Loads a library with the wide char name specified.
 * @param path Full or relative path to the plugin library
 * @return Reference to shared object
 */
// std::shared_ptr<void> load_mmap_object(const std::wstring& path);
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

}  // namespace ov
