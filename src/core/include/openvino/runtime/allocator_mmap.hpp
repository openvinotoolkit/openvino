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
#include <optional>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

/// @brief Returns the heap budget for generated constants; an unset value disables offloading.
OPENVINO_API std::optional<uint64_t> get_constant_memory_budget();

/// @brief Returns the directory holding offloaded constants. Empty means the system temporary directory.
OPENVINO_API const std::string& get_constant_offload_path();

/// @brief Decides where a constant buffer is allocated.
///
/// Returns true when the buffer must be offloaded to a file because the heap budget is exhausted.
/// Returns false when it fits, and in that case the bytes are charged against the budget until
/// release_constant_memory() is called for them.
OPENVINO_API bool should_offload_constant(const element::Type& element_type, size_t byte_size);

/// @brief Returns bytes charged by should_offload_constant() back to the budget.
OPENVINO_API void release_constant_memory(size_t byte_size);

/// @brief Applies a constant memory budget and offload path, restoring the previous ones on destruction.
///
/// An unset budget leaves the current configuration untouched; zero offloads all eligible constants.
class OPENVINO_API ScopedConstantOffloadConfig {
public:
    ScopedConstantOffloadConfig(std::optional<uint64_t> max_memory, const std::string& offload_path);
    ~ScopedConstantOffloadConfig();

    ScopedConstantOffloadConfig(const ScopedConstantOffloadConfig&) = delete;
    ScopedConstantOffloadConfig& operator=(const ScopedConstantOffloadConfig&) = delete;

private:
    std::optional<uint64_t> m_previous_budget;
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
class OPENVINO_API BudgetedHeapAllocator {
public:
    void* allocate(size_t bytes, size_t alignment);
    void deallocate(void* handle, size_t bytes, size_t alignment) noexcept;
    bool is_equal(const BudgetedHeapAllocator&) const;
};

}  // namespace ov