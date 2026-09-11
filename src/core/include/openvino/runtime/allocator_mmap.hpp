// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides file-backed mmap allocator.
 *
 * @file openvino/runtime/allocator_mmap.hpp
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/core/except.hpp"

namespace ov {

struct MMapConstantsConfig {
    bool enabled = false;
    uint64_t min_constant_size = 64ULL * 1024ULL * 1024ULL;
};

OPENVINO_API const MMapConstantsConfig& get_mmap_constants_config();

class OPENVINO_API ScopedMMapConstantsConfig {
public:
    explicit ScopedMMapConstantsConfig(const MMapConstantsConfig& config);
    ~ScopedMMapConstantsConfig();

    ScopedMMapConstantsConfig(const ScopedMMapConstantsConfig&) = delete;
    ScopedMMapConstantsConfig& operator=(const ScopedMMapConstantsConfig&) = delete;

private:
    MMapConstantsConfig m_previous_config;
};

class OPENVINO_API TemporaryFileBackedAllocator {
public:
    void* allocate(size_t bytes, size_t alignment);
    void deallocate(void* handle, size_t bytes, size_t alignment) noexcept;
    bool is_equal(const TemporaryFileBackedAllocator&) const;
};

}  // namespace ov