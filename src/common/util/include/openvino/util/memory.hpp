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

}  // namespace ov::util
