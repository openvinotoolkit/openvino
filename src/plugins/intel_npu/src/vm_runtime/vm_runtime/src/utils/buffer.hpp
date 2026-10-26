//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "allocator_core.hpp"
#include "vm_export.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace intel_npu::vm {

using BufferHandle = uint64_t;

// Logical budget cap enforced by BufferManager (sum of owned-buffer sizes), not an eager allocation.
inline constexpr uint64_t DEFAULT_BUFFER_MEMORY_LIMIT_BYTES = uint64_t{1} << 32;  // 4 GiB

enum class NPU_VM_EXPORT Permission : uint8_t {
    Read = 0,
    ReadWrite = 1,
};

enum class NPU_VM_EXPORT Ownership : uint8_t {
    Owned = 0,
    Unowned = 1,
};

class NPU_VM_EXPORT Buffer {
    // BufferManager owns the storage of Owned buffers and needs a mutable pointer to free the block and to carve
    // derived views. Granting it friendship keeps that mutable access internal -- teardown does not go through
    // const_cast -- while the public API stays read-only (getData() returns const).
    friend class BufferManager;

    uint8_t* _data = nullptr;
    size_t _size = 0;
    Permission _permission;
    Ownership _ownership;

    // Mutable view of the storage, for BufferManager-only teardown and derivation (see the friend declaration).
    uint8_t* rawData() noexcept {
        return _data;
    }

public:
    // Construct a view over an already-allocated region. BufferManager owns and releases storage for
    // buffers tagged Ownership::Owned.
    Buffer(uint8_t* data, size_t size, Permission permission, Ownership ownership = Ownership::Unowned);

    // Disable copy operations to prevent unintended shallow copies.
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // A Buffer is a non-owning view (BufferManager releases owned storage), so the destructor has nothing to release.
    // Moves transfer the view and clear the source, so moved-from objects are explicit empty views.
    // The moves and destructor are written out (rather than `= default`) so all five special members are explicitly
    // user-provided -- Coverity's rule-of-three/five checker does not count a `= default`ed member as
    // user-provided, so leaving them defaulted alongside the deleted copies trips it.
    Buffer(Buffer&& other) noexcept
            : _data(std::exchange(other._data, nullptr)),
              _size(std::exchange(other._size, 0)),
              _permission(other._permission),
              _ownership(other._ownership) {
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            _data = std::exchange(other._data, nullptr);
            _size = std::exchange(other._size, 0);
            _permission = other._permission;
            _ownership = other._ownership;
        }
        return *this;
    }
    ~Buffer() {  // NOLINT(modernize-use-equals-default): user-provided, not `= default`, per the note above
    }

    const uint8_t* getData() const;

    size_t getSize() const;

    Permission getPermission() const;

    Ownership getOwnership() const;

    void readData(size_t offset, uint8_t* data, size_t size) const;

    void writeData(size_t offset, const uint8_t* data, size_t size);

    void copyDataFrom(const Buffer& source);
};

class NPU_VM_EXPORT BufferManager {
    class BufferHandleManager {
        BufferHandle _lastHandle = 0;

    public:
        BufferHandle getNextHandle();
    };

    size_t _maximumMemorySize{};
    size_t _currentMemorySize{};
    BufferHandleManager _handleManager{};
    AnyAllocator _allocator;
    std::unordered_map<BufferHandle, Buffer> _buffers;
    std::unordered_map<BufferHandle, std::vector<BufferHandle>> _derivedBuffers;

    BufferHandle generateBufferHandle();

    // Deallocates every Owned buffer's storage through _allocator. Unowned buffers (external memory and
    // derived views) are left untouched. Does not modify the buffer maps or the budget; callers do that.
    void deallocateOwnedBuffers() noexcept;

public:
    // Create a buffer manager, which manages the creation, deletion and access for buffers
    // The buffer manager keeps track of the total allocated memory size, and does not allow creation of new buffers if
    // the total size exceeds the given maximum memory size. Owned buffers are obtained from the given allocator.
    //
    // Allocator contract: the manager reclaims Owned-buffer storage by calling allocator.deallocate() on every owned
    // block and then allocator.recycle() (in recycle()); it tracks no storage of its own. The allocator must therefore
    // release every outstanding block through that sequence. A per-block leaf reclaims each block in deallocate(); an
    // arena whose deallocate() only pops the most recent block must reclaim the remainder in recycle(). An allocator
    // that does neither would leak the dropped buffers.
    BufferManager(size_t maximumMemorySize, AnyAllocator allocator);
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    // Move assignment and the destructor must first free this object's own Owned buffers (the move
    // constructor leaves the source empty, so its later destruction frees nothing), hence they are
    // user-defined rather than defaulted.
    BufferManager(BufferManager&&) = default;
    BufferManager& operator=(BufferManager&&) noexcept;
    ~BufferManager();

    // Allocates a buffer of the given size. Returns a handle to the buffer
    // Throws std::invalid_argument if the requested size is zero
    // Throws std::bad_alloc if the requested size exceeds the maximum memory size
    BufferHandle create(size_t size, Permission permission = Permission::ReadWrite, bool zeroInitialized = false);

    // Create a buffer from external memory. Returns a handle to the created buffer
    // This buffer does not contribute to the memory size tracked by the buffer manager, as the memory
    // is external
    // Throws std::invalid_argument if the external data pointer is null or if the size is zero
    BufferHandle createFromMemory(uint8_t* externalData, size_t size, Permission permission);

    // Create a buffer from another buffer that references a subset of the original buffer's memory. Returns a handle to
    // the new buffer. The new buffer does not own the memory, and its permission is the same as the original buffer
    // Throws std::invalid_argument if the original handle is invalid
    // Throws std::out_of_range if the offset and size exceed the bounds of the original buffer
    BufferHandle createFromBuffer(BufferHandle handle, size_t offset, size_t size);

    // Deletes the buffer associated with the given handle. Does not deallocate memory for unowned buffers.
    // If the buffer has derived buffers created via createFromBuffer, those are recursively deleted first.
    // Throws std::invalid_argument if the handle is invalid
    void deleteBuffer(BufferHandle handle);

    // Returns true if a buffer with the given handle exists, false otherwise
    bool exists(BufferHandle handle) const;

    // Returns a reference to the buffer associated with the given handle
    // Throws std::invalid_argument if the handle is invalid
    Buffer& getBuffer(BufferHandle handle);

    const Buffer& getBuffer(BufferHandle handle) const;

    // Drops every buffer and rewinds the allocator's pool for reuse WITHOUT releasing it, so the next
    // inference reuses the same memory. (With an arena the per-buffer frees are no-ops and the rewind keeps
    // the memory; with a leaf the frees reclaim and the rewind is a no-op.) The handle counter stays
    // monotonic, so stale handles are not reused.
    void recycle() noexcept;

    size_t getCurrentMemorySize() const;
};

}  // namespace intel_npu::vm
