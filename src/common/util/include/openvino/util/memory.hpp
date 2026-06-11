// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

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
 * @brief Reserves a block of memory of the specified size without actually allocating physical memory.
 * The reserved memory block is not accessible until acquire_buffer() is called on it. The reserved memory block should
 * be released with release_buffer() when it is no longer needed.
 * @param size [in] Size of the memory block to reserve in bytes. Must be greater than 0.
 * @param error [out] Optional Error message in case of failure, empty otherwise.
 * @return Pointer to the reserved memory block, or nullptr if the reservation fail.
 */
void* reserve_buffer(size_t size, std::string* error = nullptr) noexcept;

/**
 * @brief Acquires the reserved memory block, making it accessible for read/write operations.
 * @param reserved_buffer [in] Pointer to the reserved memory block to acquire.
 * @param size [in] Size of the memory block to acquire in bytes. Must be greater than 0.
 * @param error [out] Optional Error message in case of failure, empty otherwise.
 * @note The reserved memory block should be successfully acquired before it can be used.
 */
void acquire_buffer(void* reserved_buffer, size_t size, std::string* error = nullptr) noexcept;

/**
 * @brief Evicts the acquired memory block, making it inaccessible and allowing the system to free physical memory.
 * @param reserved_buffer [in] Pointer to the reserved memory block to evict.
 * @param size [in] Size of the memory block to evict in bytes. Must be greater than 0.
 * @param error [out] Optional Error message in case of failure, empty otherwise.
 * @note After eviction, the reserved memory block can be acquired again with acquire_buffer().
 */
void evict_buffer(void* reserved_buffer, size_t size, std::string* error = nullptr) noexcept;

/**
 * @brief Releases the reserved memory block, freeing any associated resources. After this call, the reserved memory
 * block is no longer valid and should not be used.
 * @param reserved_buffer [in] Pointer to the reserved memory block to release.
 * @param size [in] Size of the memory block to release in bytes. Must be greater than 0.
 * @param error [out] Optional Error message in case of failure, empty otherwise.
 */
void release_buffer(void* reserved_buffer, size_t size, std::string* error = nullptr) noexcept;

}  // namespace ov::util
