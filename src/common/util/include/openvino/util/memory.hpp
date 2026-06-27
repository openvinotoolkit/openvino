// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>

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
 * @brief Rounds @p size down to the nearest multiple of @p alignment.
 *
 * @param alignment  Alignment boundary. Must be a power of two and greater than zero.
 * @return Largest value <= @p size that is a multiple of @p alignment.
 */
constexpr size_t align_size_down(size_t size, size_t alignment) noexcept {
    return size & ~(alignment - 1);
}

/** @brief Represents a memory region aligned to a power-of-two boundary. */
struct AlignedRegion {
    uintptr_t m_address = 0;  //!< Aligned base address (rounded down to boundary)
    size_t m_length = 0;      //!< Total length of the aligned region including the gap
    size_t m_gap = 0;         //!< Gap from the aligned address to the original unaligned address
};

/**
 * @brief Aligns a memory region to a power-of-two boundary (rounded down).
 *
 * Computes the largest aligned address <= @p base and the gap between that
 * aligned address and @p base, returning a region large enough to cover
 * [base, base + raw_len).
 *
 * @param base      The original (potentially unaligned) base address.
 * @param raw_len   The length of the region starting at @p base.
 * @param alignment The alignment boundary. Must be a power of two and greater than zero.
 * @return AlignedRegion covering at least [base, base + raw_len).
 */
constexpr AlignedRegion align_region(uintptr_t base, size_t raw_len, size_t alignment) noexcept {
    const auto aligned = base & ~(static_cast<uintptr_t>(alignment) - 1);
    const auto gap = static_cast<size_t>(base - aligned);
    return {aligned, raw_len + gap, gap};
}

/**
 * @brief Allocates @p size bytes of uninitialized memory on the specified @p alignment boundary.
 *
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

/**
 * @brief Reserves virtual address space of the given size without backing it with physical memory.
 * The region is inaccessible until vm_commit() is called. Release with vm_release() when no longer needed.
 * @param size  Size in bytes to reserve. Must be greater than 0.
 * @param ec    Set to the OS error code on failure, cleared on success.
 * @return Pointer to the reserved region, or nullptr on failure.
 */
void* vm_reserve(size_t size, std::error_code& ec) noexcept;

/**
 * @brief Commits a previously reserved region, making it readable and writable.
 * @param ptr   Pointer returned by vm_reserve().
 * @param size  Size in bytes to commit. Must be greater than 0.
 * @param ec    Set to the OS error code on failure, cleared on success.
 */
void vm_commit(void* ptr, size_t size, std::error_code& ec) noexcept;

/**
 * @brief Decommits a committed region: revokes access and returns physical pages to the OS.
 * The virtual address range remains reserved and can be committed again with vm_commit().
 * @param ptr   Pointer returned by vm_reserve(). Must not be nullptr.
 * @param size  Size in bytes to decommit. Must be greater than 0.
 * @pre  ptr != nullptr && size > 0; violated preconditions are a programming error (assert fires in debug).
 */
void vm_decommit(void* ptr, size_t size) noexcept;

/**
 * @brief Releases the reserved virtual address range. Can be called without a prior vm_decommit().
 * After this call the pointer is invalid and must not be used.
 * @param ptr   Pointer returned by vm_reserve(). Must not be nullptr.
 * @param size  Size in bytes originally passed to vm_reserve(). Must be greater than 0.
 * @pre  ptr != nullptr && size > 0; violated preconditions are a programming error (assert fires in debug).
 */
void vm_release(void* ptr, size_t size) noexcept;

/**
 * @brief Pre-fetch a committed VM range into physical memory.
 *
 * Works with both anonymous (@ref vm_commit) and file-backed (mmap) regions.
 *
 * @param ptr         Base address of the range. Must be page-aligned.
 * @param size        Number of bytes to pre-fetch. Must be a multiple of the system page size.
 * @param num_threads Strategy selector:
 *                    - @c 0 (default) → OS advisory hint (async, low overhead).
 *                    - @c N >= 1      → parallel touch with N threads (synchronous).
 */
void vm_prefetch(void* ptr, size_t size, size_t num_threads = 0) noexcept;

}  // namespace ov::util
