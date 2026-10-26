//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffer.hpp"
#include "allocator_core.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <stdexcept>
#include <utility>

intel_npu::vm::Buffer::Buffer(uint8_t* data, size_t size, Permission permission, Ownership ownership)
        : _data(data), _size(size), _permission(permission), _ownership(ownership) {
}

const uint8_t* intel_npu::vm::Buffer::getData() const {
    return _data;
}

size_t intel_npu::vm::Buffer::getSize() const {
    return _size;
}

intel_npu::vm::Permission intel_npu::vm::Buffer::getPermission() const {
    return _permission;
}

intel_npu::vm::Ownership intel_npu::vm::Buffer::getOwnership() const {
    return _ownership;
}

void intel_npu::vm::Buffer::readData(size_t offset, uint8_t* data, size_t size) const {
    if (size == 0) {
        return;
    }
    if (_data == nullptr) {
        throw std::runtime_error("Buffer data pointer is null");
    }
    if (_permission != Permission::ReadWrite && _permission != Permission::Read) {
        throw std::runtime_error("Buffer does not have read permission");
    }
    if (offset > _size || size > _size - offset) {
        throw std::out_of_range("Read exceeds buffer bounds");
    }
    std::memcpy(data, std::next(_data, static_cast<std::ptrdiff_t>(offset)), size);
}

void intel_npu::vm::Buffer::writeData(size_t offset, const uint8_t* data, size_t size) {
    // std::memcpy takes size_t, so copies whose size does not fit in size_t cannot be performed on this host.
    static_assert(sizeof(size_t) >= sizeof(int64_t), "Bytecode buffer copies require a 64-bit size_t");
    if (size == 0) {
        return;
    }
    if (_data == nullptr) {
        throw std::runtime_error("Buffer data pointer is null");
    }
    if (_permission != Permission::ReadWrite) {
        throw std::runtime_error("Buffer does not have write permission");
    }
    if (offset > _size || size > _size - offset) {
        throw std::out_of_range("Write exceeds buffer bounds");
    }
    std::memcpy(std::next(_data, static_cast<std::ptrdiff_t>(offset)), data, size);
}

void intel_npu::vm::Buffer::copyDataFrom(const Buffer& source) {
    if (_permission != Permission::ReadWrite) {
        throw std::runtime_error("Destination buffer does not have write permission");
    }
    if (source._permission != Permission::ReadWrite && source._permission != Permission::Read) {
        throw std::runtime_error("Source buffer does not have read permission");
    }
    if (_size != source._size) {
        throw std::invalid_argument("Buffer copy requires equal buffer sizes");
    }
    if (_size == 0) {
        return;
    }
    if (_data == nullptr || source._data == nullptr) {
        throw std::runtime_error("Buffer data pointer is null");
    }
    std::memmove(_data, source._data, _size);
}

intel_npu::vm::BufferHandle intel_npu::vm::BufferManager::BufferHandleManager::getNextHandle() {
    return ++_lastHandle;
}

intel_npu::vm::BufferHandle intel_npu::vm::BufferManager::generateBufferHandle() {
    return _handleManager.getNextHandle();
}

void intel_npu::vm::BufferManager::deallocateOwnedBuffers() noexcept {
    if (!_allocator) {
        return;  // moved-from manager: _buffers is empty, nothing to free
    }
    for (auto& entry : _buffers) {
        auto& buffer = entry.second;
        if (buffer.getOwnership() == Ownership::Owned) {
            _allocator.deallocate(Block{buffer.rawData(), buffer.getSize()});
        }
    }
}

void intel_npu::vm::BufferManager::recycle() noexcept {
    // Free the Owned-buffer storage, then let the allocator reclaim its pool (see recycle() in buffer.hpp).
    deallocateOwnedBuffers();
    if (_allocator) {
        _allocator.recycle();
    }
    _buffers.clear();
    _derivedBuffers.clear();
    _currentMemorySize = 0;
}

intel_npu::vm::BufferManager::BufferManager(size_t maximumMemorySize, AnyAllocator allocator)
        : _maximumMemorySize(maximumMemorySize), _allocator(std::move(allocator)) {
    if (!_allocator) {
        throw std::invalid_argument("BufferManager: allocator must not be null");
    }
}

intel_npu::vm::BufferManager& intel_npu::vm::BufferManager::operator=(BufferManager&& other) noexcept {
    if (this != &other) {
        // Free our current Owned buffers before overwriting members: deallocateOwnedBuffers() reads the
        // current _allocator/_buffers, so it must run before _allocator is moved out.
        deallocateOwnedBuffers();
        _maximumMemorySize = other._maximumMemorySize;
        _currentMemorySize = other._currentMemorySize;
        _handleManager = other._handleManager;
        _allocator = std::move(other._allocator);
        // Moving the maps destroys the just-freed view entries (no double free) and adopts other's buffers.
        _buffers = std::move(other._buffers);
        _derivedBuffers = std::move(other._derivedBuffers);
    }
    return *this;
}

intel_npu::vm::BufferManager::~BufferManager() {
    // Free the storage of any Owned buffers still held (see deallocateOwnedBuffers).
    deallocateOwnedBuffers();
}

intel_npu::vm::BufferHandle intel_npu::vm::BufferManager::create(size_t size, Permission permission,
                                                                 bool zeroInitialized) {
    if (size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }
    if (size > _maximumMemorySize || _currentMemorySize > _maximumMemorySize - size) {
        throw std::bad_alloc();
    }

    const auto block = _allocator.allocate(size);
    if (!block) {
        throw std::bad_alloc();
    }

    // By default, owned buffers are zero-initialized so recycled memory never exposes bytes from a previous execution.
    // Zeroing out the allocated memory causes significant performance overhead for large buffers.
    // Callers may opt out (zeroInitialized=true) only when the buffer is guaranteed to be initialized
    if (!zeroInitialized) {
        std::memset(block.ptr, 0, block.size);
    }

    const auto bufferHandle = generateBufferHandle();
    try {
        _buffers.emplace(bufferHandle, Buffer(block.ptr, block.size, permission, Ownership::Owned));
    } catch (...) {
        _allocator.deallocate(block);
        throw;
    }
    _currentMemorySize += size;
    return bufferHandle;
}

intel_npu::vm::BufferHandle intel_npu::vm::BufferManager::createFromMemory(uint8_t* externalData, size_t size,
                                                                           Permission permission) {
    if (externalData == nullptr) {
        throw std::invalid_argument("External data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }
    const auto handle = generateBufferHandle();
    _buffers.emplace(handle, Buffer(externalData, size, permission, Ownership::Unowned));
    return handle;
}

intel_npu::vm::BufferHandle intel_npu::vm::BufferManager::createFromBuffer(intel_npu::vm::BufferHandle handle,
                                                                           size_t offset, size_t size) {
    auto& buffer = this->getBuffer(handle);
    const auto bufferSize = buffer.getSize();
    if (offset > bufferSize || size > bufferSize - offset) {
        throw std::out_of_range("Requested offset and size exceeds buffer bounds");
    }
    const auto newData = std::next(buffer.rawData(), static_cast<std::ptrdiff_t>(offset));
    const auto newHandle = generateBufferHandle();
    _buffers.emplace(newHandle, Buffer(newData, size, buffer.getPermission(), Ownership::Unowned));
    _derivedBuffers[handle].push_back(newHandle);
    return newHandle;
}

void intel_npu::vm::BufferManager::deleteBuffer(intel_npu::vm::BufferHandle handle) {
    // Recursively delete all buffers derived from this one via createFromBuffer.
    // Children that were already explicitly deleted are skipped.
    auto childrenIt = _derivedBuffers.find(handle);
    if (childrenIt != _derivedBuffers.end()) {
        // Copy the children list before iterating, as each recursive call modifies _derivedBuffers
        const auto children = childrenIt->second;
        for (const auto child : children) {
            if (exists(child)) {
                deleteBuffer(child);
            }
        }
        _derivedBuffers.erase(handle);
    }

    auto& buffer = this->getBuffer(handle);
    if (buffer.getOwnership() == Ownership::Owned) {
        _currentMemorySize -= buffer.getSize();
        _allocator.deallocate(Block{buffer.rawData(), buffer.getSize()});
    }
    _buffers.erase(handle);
}

bool intel_npu::vm::BufferManager::exists(intel_npu::vm::BufferHandle handle) const {
    return _buffers.find(handle) != _buffers.end();
}

intel_npu::vm::Buffer& intel_npu::vm::BufferManager::getBuffer(intel_npu::vm::BufferHandle handle) {
    return const_cast<Buffer&>(std::as_const(*this).getBuffer(handle));
}

const intel_npu::vm::Buffer& intel_npu::vm::BufferManager::getBuffer(intel_npu::vm::BufferHandle handle) const {
    auto it = _buffers.find(handle);
    if (it == _buffers.end()) {
        throw std::invalid_argument("Invalid buffer handle");
    }
    return it->second;
}

size_t intel_npu::vm::BufferManager::getCurrentMemorySize() const {
    return _currentMemorySize;
}
