// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdlib>

namespace ov::util {

/**
 * @brief Rounds @p size up to the nearest multiple of @p alignment.
 *
 * @param size       Value to round up.
 * @param alignment  Alignment boundary. Must be a power of two and greater than zero.
 * @return Smallest value >= @p size that is a multiple of @p alignment.
 */
constexpr size_t align_size_up(size_t size, size_t alignment) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Allocates @p size bytes of uninitialized memory on the specified @p alignment boundary.
 *
 * Cross-platform wrapper: for aligned memory allocation and deallocation.
 *
 * @param size       Number of bytes to allocate. Must be greater than zero.
 * @param alignment  Desired alignment in bytes. Must be a power of two.
 *                   Passing `0` applies no specific alignment constraint (`alignof(std::max_align_t)` is used).
 * @return Pointer to the allocated memory, or `nullptr` on failure.
 */
void* aligned_alloc(size_t size, size_t alignment) noexcept;

/**
 * @brief Releases memory previously allocated by @ref aligned_alloc.
 *
 * @param ptr  Pointer returned by @ref aligned_alloc. Passing `nullptr` is a no-op.
 */
void aligned_free(void* ptr) noexcept;

}  // namespace ov::util
