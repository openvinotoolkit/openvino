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

std::shared_ptr<ngraph::runtime::AlignedBuffer> load_mmap_object(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<ngraph::runtime::AlignedBuffer> load_mmap_object(const std::wstring& path);

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

}  // namespace ov
