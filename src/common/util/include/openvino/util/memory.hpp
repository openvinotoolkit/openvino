// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <string>
#include <system_error>
#include <vector>

namespace ov::util {

/** @brief Minimum guaranteed page alignment on all supported platforms (x86, ARM, RISC-V). */
inline constexpr size_t min_page_alignment = 4096;

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
 * @param ptr   Base address of the range. Must be page-aligned.
 * @param size  Number of bytes to pre-fetch. Must be a multiple of the system page size.
 * @param fast  Strategy selector:
 *              - @c true (default) → OS advisory hint (async, low overhead).
 *              - @c false          → parallel synchronous touch that blocks until every page is
 *                resident. The degree of parallelism is chosen internally (see the shared pool
 *                used by @ref vm_prefetch_async), not by the caller.
 */
void vm_prefetch(void* ptr, size_t size, bool fast = true) noexcept;

/**
 * @brief Move-only RAII token representing background page-population work started by
 * @ref vm_prefetch_async.
 *
 * The work itself runs on a shared, bounded background thread pool (see @ref vm_prefetch_async):
 * the token owns the `std::future`s of the submitted tasks, not dedicated OS threads. Call
 * wait() to block until the population completes, or simply let the token go out of scope —
 * its destructor waits for all outstanding tasks, so no background work is ever left running
 * uncontrolled.
 *
 * A default-constructed (or moved-from, or detached) token is "empty": wait() is a no-op and
 * valid()/operator bool() return false.
 *
 * @note The token does not extend the lifetime of the underlying memory mapping/buffer that
 * is being populated. The caller is responsible for keeping that memory alive until the
 * token has completed (via wait() or destruction) — unless detach() is used, see below.
 */
class PrefetchToken {
public:
    /**
     * @brief Callback used by detach() to hand off the still-running tasks to another owner
     * (typically the object whose memory is being populated, e.g. a `MappedMemory`
     * implementation), which then becomes responsible for waiting on them before it destroys
     * the underlying memory. Set via set_detach_sink() by whoever creates the token.
     */
    using DetachSink = std::function<void(std::vector<std::future<void>>&&)>;

    PrefetchToken() noexcept = default;
    explicit PrefetchToken(std::vector<std::future<void>>&& tasks) noexcept : m_tasks(std::move(tasks)) {}

    PrefetchToken(const PrefetchToken&) = delete;
    PrefetchToken& operator=(const PrefetchToken&) = delete;

    PrefetchToken(PrefetchToken&&) noexcept = default;

    PrefetchToken& operator=(PrefetchToken&& other) noexcept {
        if (this != &other) {
            wait();  // avoid abandoning still-running tasks owned by *this via member assignment
            m_tasks = std::move(other.m_tasks);
            m_detach_sink = std::move(other.m_detach_sink);
            other.m_detach_sink = nullptr;
        }
        return *this;
    }

    ~PrefetchToken() {
        wait();
    }

    /**
     * @brief Blocks until all background population tasks complete, then releases them.
     * Safe to call multiple times and on an empty token (no-op).
     */
    void wait() {
        for (auto& task : m_tasks) {
            if (task.valid()) {
                task.wait();
            }
        }
        m_tasks.clear();
        m_detach_sink = nullptr;
    }

    /**
     * @brief Registers a sink that will receive the outstanding tasks when detach() is called.
     * Intended for use by the code that creates the token (e.g. a `MappedMemory`
     * implementation), so it can keep the tasks alive internally and join them itself (for
     * example in its own destructor). Not intended to be called by generic client code.
     */
    void set_detach_sink(DetachSink sink) noexcept {
        m_detach_sink = std::move(sink);
    }

    /**
     * @brief Detaches the background tasks from this token: the token's destructor will no
     * longer wait for them.
     *
     * If a detach sink was registered (via set_detach_sink()) — which is the case for tokens
     * returned by `MappedMemory::hint_prefetch_async()` — ownership of the outstanding tasks is
     * handed off to that sink, which keeps them alive and waits for them at an appropriate
     * point (typically when the memory being populated is destroyed). This is the safe way to
     * use detach(): the memory stays valid until the background work is guaranteed to be done.
     *
     * If no sink was registered, the tasks are simply released without waiting, mirroring
     * `std::thread::detach()` — the caller then takes on full responsibility for making sure
     * the populated memory outlives the background work.
     *
     * After detach(), the token is empty (valid() == false).
     */
    void detach() noexcept {
        if (m_detach_sink) {
            m_detach_sink(std::move(m_tasks));
        }
        m_tasks.clear();
        m_detach_sink = nullptr;
    }

    /** @brief Returns true if the token owns outstanding background work. */
    bool valid() const noexcept {
        return !m_tasks.empty();
    }

    explicit operator bool() const noexcept {
        return valid();
    }

private:
    std::vector<std::future<void>> m_tasks;
    DetachSink m_detach_sink;
};

/**
 * @brief Asynchronous variant of @ref vm_prefetch.
 *
 * Starts pre-fetching a committed VM range into physical memory by submitting page-touching
 * tasks to a small, shared background thread pool, and returns immediately with a
 * @ref PrefetchToken that must be used to wait for completion (explicitly via wait(), or
 * implicitly by letting the token go out of scope).
 *
 * Unlike spawning dedicated threads per call, repeated calls reuse the same bounded set of
 * pool worker threads: tasks queue up and are picked up by whichever worker becomes free.
 *
 * @param ptr         Base address of the range. Must be page-aligned.
 * @param size        Number of bytes to pre-fetch. Must be a multiple of the system page size.
 * @return A @ref PrefetchToken owning the submitted tasks' futures (empty when @p num_threads ==
 * 0).
 */
PrefetchToken vm_prefetch_async(void* ptr, size_t size) noexcept;

}  // namespace ov::util
