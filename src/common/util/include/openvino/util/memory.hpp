// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>

namespace ov::util {

/** @brief One mebibyte (1024 * 1024 bytes). */
inline constexpr size_t one_mib = 1024 * 1024;

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
 * @brief Queryable facts about a memory buffer's allocation, set once at construction/mapping time.
 */
struct MemoryProperties {
    /// @brief Opaque id of the buffer's ultimate backing allocation (e.g. weight-sharing/mmap root). 0 == unknown.
    size_t source_id = 0;
    /// @brief This buffer's byte offset within the source_id allocation. Meaningless when source_id == 0.
    size_t offset = 0;
};

/// @brief Read-only, non-owning view (pointer + size) of a buffer's contents.
class MemoryView {
public:
    constexpr MemoryView() noexcept = default;
    constexpr MemoryView(const std::byte* data, size_t size) noexcept : m_data{data}, m_size{size} {}

    constexpr const std::byte* data() const noexcept {
        return m_data;
    }
    constexpr size_t size() const noexcept {
        return m_size;
    }
    constexpr const std::byte* begin() const noexcept {
        return data();
    }
    constexpr const std::byte* end() const noexcept {
        return data() + size();
    }

private:
    const std::byte* m_data = nullptr;
    size_t m_size = 0;
};

/**
 * @brief Common, read-only access to a contiguous block of memory plus its properties.
 */
class IBuffer {
public:
    // Declared first so new virtual methods always append after it, keeping this slot stable.
    virtual ~IBuffer() = default;

    virtual const void* data() const noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual const MemoryProperties& get_properties() const noexcept = 0;

    /// @brief Read-only view for bulk reads/copies (e.g. memcpy, std::copy); not virtual, built on data()/size().
    MemoryView view() const noexcept {
        return {static_cast<const std::byte*>(data()), size()};
    }

    /// @brief Hint to release the underlying memory if possible (e.g. unmaps/decommits); no-op by default.
    virtual void hint_evict() noexcept {}
    /**
     * @brief Hint to fetch the data to memory. No-op by default.
     */
    virtual void hint_prefetch() const {}

    /// @brief Buffer this one is a view into (see MemoryProperties::source_id/offset), if any. Default: none.
    virtual std::shared_ptr<IBuffer> get_source_buffer() const {
        return nullptr;
    }
};

/**
 * @brief Extends IBuffer with mutable access.
 */
class IMutableBuffer : public IBuffer {
public:
    using IBuffer::data;
    virtual void* data() noexcept = 0;
};

/**
 * @brief CRTP mixin adding typed pointer-cast convenience (data_as<T>()) to a type exposing data()/data()
 * const.
 */
template <typename Derived>
class TypedMemoryAccessor {
public:
    template <typename T>
    constexpr const T* data_as() const noexcept {
        return reinterpret_cast<const T*>(static_cast<const Derived*>(this)->Derived::data());
    }
    template <typename T>
    constexpr T* data_as() noexcept {
        return reinterpret_cast<T*>(static_cast<Derived*>(this)->Derived::data());
    }
};

}  // namespace ov::util
