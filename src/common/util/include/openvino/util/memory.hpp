// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
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
 * @brief Rounds @p size down to the nearest multiple of @p alignment.
 *
 * @param size       Value to round down.
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
 * @brief Checks if the memory at @p data was allocated via mmap (or the platform equivalent).
 *
 * On Windows, uses VirtualQuery: MEM_MAPPED regions (MapViewOfFile / MapViewOfFile3)
 * return true; MEM_PRIVATE regions (heap, VirtualAlloc) return false.
 *
 * On Linux, returns true for any non-null pointer: the NPU Level Zero driver
 * accepts all memory types as a standard OS allocation on this platform.
 *
 * @param data  Pointer to query. Must not be null.
 * @return true if the memory was allocated via mmap (or its platform equivalent).
 */
bool is_mmap_memory(const void* data) noexcept;

/**
 * @brief Checks that the entire range [data, data + size) lies within a single file-backed
 *        mmap region — i.e. is not split across independently allocated views.
 *
 * On Windows, the placeholder-based mmap creates the file-backed portion as a SINGLE view
 * (one AllocationBase for the whole file content).  When the file size is not a multiple of
 * 64 KB (the system allocation granularity), the VA range is extended with an anonymous
 * pagefile-backed tail section that has a different AllocationBase.  A blob whose range
 * [data, data+size) extends into this tail cannot be imported with
 * ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT because the driver cannot access those bytes
 * through the original file section.
 *
 * The AllocationBase discriminator also correctly handles subsequent eviction: hint_evict()
 * unmaps and re-splits the single view into independently re-mapped pieces, each with a
 * distinct AllocationBase.  After any eviction the check returns false, preventing
 * a persistent import of a fragmented mapping.
 *
 * On Linux, mmap always produces a single contiguous file-backed region with no
 * anonymous padding, so only the start pointer is checked.
 *
 * @param data  Start of the range to check.
 * @param size  Length of the range in bytes. Must be > 0.
 * @return true if [data, data + size) is entirely within one contiguous file-backed mmap
 *         allocation (same AllocationBase at start and end, both MEM_MAPPED).
 */
bool is_single_mmap_region(const void* data, size_t size) noexcept;

}  // namespace ov::util
