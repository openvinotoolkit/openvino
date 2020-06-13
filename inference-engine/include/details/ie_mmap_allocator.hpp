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

#ifdef _WIN32
#else
# include <unistd.h>
# include <fcntl.h>
# include <sys/mman.h>
# include <sys/stat.h>
#endif

namespace InferenceEngine {
namespace details {
/*
 * @brief This is a helper class to wrap memory mapped files
 */
class MmapAllocator : public IAllocator {
public:
    using Ptr = std::shared_ptr<MmapAllocator>;

    MmapAllocator(const char* path, size_t offset = 0, LockOp lock = LOCK_FOR_READ)
        : _path(path), _offset(offset), _size(0), _lock(lock), _data(nullptr), _file(0) {
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
        map(_path.c_str(), _offset, size, _lock);
        return _data;
    }
    /**
     * @brief The PreAllocator class cannot release the handle
     * @return false
     */
    bool free(void*) noexcept override {  // NOLINT
        unmap();
        return true;
    }

    /**
     * @brief Deletes current allocator.
     * Can be used if a shared_from_irelease pointer is used
     */
    void Release() noexcept override {
        free(nullptr);
        delete this;
    }

    size_t size() {
        return _size;
    }

protected:
    virtual ~MmapAllocator() = default;

    void map(const char* path, size_t offset, size_t size, LockOp lock) {
#ifdef _WIN32
#else
        int prot;
        int mode;

        switch (lock) {
            case LOCK_FOR_READ:
                prot = PROT_READ;
                mode = O_RDONLY;
                break;
            case LOCK_FOR_WRITE:
                prot = PROT_WRITE;
                mode = O_WRONLY;
                break;
            default:
                THROW_IE_EXCEPTION <<  "Unsupported lock option.";
        }

        struct stat sb;
        _file = open(path, mode);

        if (_file == -1) {
            THROW_IE_EXCEPTION << "Can not open file for mapping.";
        }

        if (fstat(_file, &sb) == -1) {
            close(_file);
            THROW_IE_EXCEPTION << "Can not get file size.";
        }
        
        size_t file_size = (size_t)sb.st_size;

        if (size != 0 && size > file_size) {
            close(_file);
            THROW_IE_EXCEPTION << "File size is less than requested map size.";
        }

        _size = (size == 0) ? (size_t)sb.st_size : size;

        _data = mmap(NULL, _size, prot, MAP_PRIVATE, _file, offset);
    #endif
    }

    void unmap() {
        if (_data != nullptr) {
            munmap(_data, _size);
            _data = nullptr;
            _size = 0;
        }

        if (_file != 0) {
            close(_file);
            _file = 0;
        }
    }

private:
    std::string _path;
    size_t _offset;
    size_t _size;
    LockOp _lock;

    void* _data;
#ifdef _WIN32
#else
    int _file;
#endif
};

/**
 * @brief Creates a special allocator that only works on external memory
 * @param ptr Pointer to preallocated memory
 * @param size Number of elements allocated
 * @return A new allocator
 */
std::shared_ptr<IAllocator> make_mmap_allocator(const char* path, size_t offset = 0, LockOp lock = LOCK_FOR_READ) {
    return shared_from_irelease(new MmapAllocator(path, offset, lock));
}

}  // namespace details
}  // namespace InferenceEngine
