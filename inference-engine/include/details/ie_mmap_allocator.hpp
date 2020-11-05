// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file defines utility PreAllocator class
 * 
 * @file ie_pre_allocator.hpp
 */
#pragma once

#include <details/ie_exception.hpp>
#include <memory>

#include "ie_allocator.hpp"
#include  <mmap/ie_mmap.hpp>

namespace InferenceEngine {
namespace details {
/*
 * @brief This is a helper class to wrap memory mapped files
 */
class MmapAllocator : public IAllocator {
public:
    using Ptr = std::shared_ptr<MmapAllocator>;

    MmapAllocator(const path_type& path, size_t offset = 0, LockOp lock = LOCK_FOR_READ)
        : _path(path), _size(0), _offset(offset), _lock(lock) {
    }

    /**
     * @brief Locks a handle to heap memory accessible by any memory manipulation routines
     * @return The generic pointer to a memory buffer
     */
    void* lock(void* handle, LockOp) noexcept override {
        return handle;
    }
    /**
     * @brief The MmapAllocator class does not utilize this function
     */
    void unlock(void*) noexcept override {}  // NOLINT
    /**
     * @brief Returns a pointer to mapped memory
     * @param size Size in bytes
     * @return A handle to the preallocated memory or nullptr
     */
    void* alloc(size_t size) noexcept override {
        _mmap = make_mmap(_path, size, _offset, _lock);
        return _mmap->data();
    }
    /**
     * @brief The PreAllocator class cannot release the handle
     * @return false
     */
    bool free(void*) noexcept override {  // NOLINT
        _mmap = nullptr;
        return true;
    }
    /**
     * @brief Deletes current allocator.
     * Can be used if a shared_from_irelease pointer is used
     */
    void Release() noexcept override {
        _mmap = nullptr;
        delete this;
    }
    /**
     * @brief Returns size of allocated  memory.
     */
    size_t size() {
        return _mmap->size();
    }

protected:
    virtual ~MmapAllocator() = default;

private:
    path_type _path;
    size_t _size;
    size_t _offset;
    LockOp _lock;

    IMmap::Ptr  _mmap;
};

/**
 * @brief Creates a memory mapped file allocator
 * @param path Path to a file
 * @param size Size to map, 0 to entire file size
 * @param offset Offset from beggining of file to map
 * @param lock Map mode
 * @return A new allocator
 */

std::shared_ptr<IAllocator> make_mmap_allocator(const path_type& path, size_t offset = 0, LockOp lock = LOCK_FOR_READ) {
    return shared_from_irelease(new MmapAllocator(path, offset, lock));
}

}  // namespace details
}  // namespace InferenceEngine
