// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

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

/**
 * @brief Move-only RAII token representing background page-population work started by
 * @ref vm_prefetch_async.
 *
 * The token owns the worker threads spawned to touch/populate pages. Call wait() to block
 * until the population completes, or simply let the token go out of scope — its destructor
 * joins all outstanding threads, so no background work is ever left running uncontrolled.
 *
 * A default-constructed (or moved-from) token is "empty": wait() is a no-op and valid()/
 * operator bool() return false.
 *
 * @note The token does not extend the lifetime of the underlying memory mapping/buffer that
 * is being populated. The caller is responsible for keeping that memory alive until the
 * token has completed (via wait() or destruction).
 */
class PrefetchToken {
public:
    PrefetchToken() noexcept = default;
    explicit PrefetchToken(std::vector<std::thread>&& threads) noexcept : m_threads(std::move(threads)) {}

    PrefetchToken(const PrefetchToken&) = delete;
    PrefetchToken& operator=(const PrefetchToken&) = delete;

    PrefetchToken(PrefetchToken&&) noexcept = default;

    PrefetchToken& operator=(PrefetchToken&& other) noexcept {
        if (this != &other) {
            wait();  // avoid destroying still-joinable threads owned by *this via vector assignment
            m_threads = std::move(other.m_threads);
        }
        return *this;
    }

    ~PrefetchToken() {
        wait();
    }

    /**
     * @brief Blocks until all background population threads complete, then releases them.
     * Safe to call multiple times and on an empty token (no-op).
     */
    void wait() {
        for (auto& t : m_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        m_threads.clear();
    }

    /** @brief Returns true if the token owns outstanding background work. */
    bool valid() const noexcept {
        return !m_threads.empty();
    }

    explicit operator bool() const noexcept {
        return valid();
    }

private:
    std::vector<std::thread> m_threads;
};

/**
 * @brief Asynchronous variant of @ref vm_prefetch.
 *
 * Starts pre-fetching a committed VM range into physical memory in background threads and
 * returns immediately with a @ref PrefetchToken that must be used to wait for completion
 * (explicitly via wait(), or implicitly by letting the token go out of scope).
 *
 * @param ptr         Base address of the range. Must be page-aligned.
 * @param size        Number of bytes to pre-fetch. Must be a multiple of the system page size.
 * @param num_threads Number of worker threads to spawn:
 *                    - @c 0 (default) → OS advisory hint is issued synchronously (cheap, returns
 *                      an empty token since no background threads are needed).
 *                    - @c N >= 1      → N background threads are spawned to touch pages in
 *                      parallel; the returned token owns them.
 * @return A @ref PrefetchToken owning the spawned worker threads (empty when @p num_threads == 0).
 */
PrefetchToken vm_prefetch_async(void* ptr, size_t size, size_t num_threads = 0) noexcept;

}  // namespace ov::util
