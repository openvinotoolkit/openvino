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
#include <filesystem>
#include <optional>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

/// @brief Returns whether offloading is enabled for the current read/compile scope.
OPENVINO_API bool is_constant_offload_enabled();

/// @brief Returns the directory holding offloaded constants. Empty means the system temporary directory.
OPENVINO_API std::string get_constant_offload_path();

/// @brief Reserves bytes against the active total offload size limit; throws if the limit would be exceeded.
OPENVINO_API uint64_t reserve_constant_offload(size_t byte_size);

/// @brief Releases bytes previously reserved by reserve_constant_offload().
OPENVINO_API void release_constant_offload(size_t byte_size, uint64_t generation) noexcept;

/// @brief Enables constant offloading for a scope and restores the previous configuration on destruction.
class OPENVINO_API ScopedConstantOffloadConfig {
public:
    ScopedConstantOffloadConfig(std::optional<std::filesystem::path> path, uint64_t max_size);
    ~ScopedConstantOffloadConfig();

    ScopedConstantOffloadConfig(const ScopedConstantOffloadConfig&) = delete;
    ScopedConstantOffloadConfig& operator=(const ScopedConstantOffloadConfig&) = delete;

private:
    bool m_previous_enabled{};
    uint64_t m_previous_max_size{};
    std::string m_previous_path;
    bool m_active;
};

class OPENVINO_API TemporaryFileBackedAllocator {
public:
    void* allocate(size_t bytes, size_t alignment);
    void deallocate(void* handle, size_t bytes, size_t alignment) noexcept;
    bool is_equal(const TemporaryFileBackedAllocator&) const;
};

/// @brief Heap allocator that returns its bytes to the constant memory budget when freed.
}  // namespace ov