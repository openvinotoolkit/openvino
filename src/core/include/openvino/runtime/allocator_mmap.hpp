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
#include "openvino/core/type/element_type.hpp"

namespace ov {

/// @brief Returns the minimum constant size, in bytes, offloaded to temporary files. Zero disables offloading.
OPENVINO_API uint64_t get_constant_offload_min_size();

/// @brief Tells whether a constant buffer of the given type and size should be offloaded to a temporary file.
OPENVINO_API bool should_offload_constant(const element::Type& element_type, size_t byte_size);

/// @brief Applies a constant offload threshold to the calling thread and restores the previous one on destruction.
class OPENVINO_API ScopedConstantOffloadConfig {
public:
    explicit ScopedConstantOffloadConfig(uint64_t min_constant_size);
    ~ScopedConstantOffloadConfig();

    ScopedConstantOffloadConfig(const ScopedConstantOffloadConfig&) = delete;
    ScopedConstantOffloadConfig& operator=(const ScopedConstantOffloadConfig&) = delete;

private:
    uint64_t m_previous_min_constant_size;
    bool m_enables_offload;
};

class OPENVINO_API TemporaryFileBackedAllocator {
public:
    void* allocate(size_t bytes, size_t alignment);
    void deallocate(void* handle, size_t bytes, size_t alignment) noexcept;
    bool is_equal(const TemporaryFileBackedAllocator&) const;
};

}  // namespace ov