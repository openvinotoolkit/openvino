// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared memory map objects
 * @file mmap_object.hpp
 */

#pragma once

#include <memory>
#include <string>

namespace ov {

struct MappedMemory {
    virtual char* data() noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual ~MappedMemory() = default;
};

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::string& path,
                                                   const size_t data_size = 0,
                                                   const size_t offset = 0);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::wstring& path,
                                                   const size_t data_size = 0,
                                                   const size_t offset = 0);

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

}  // namespace ov
