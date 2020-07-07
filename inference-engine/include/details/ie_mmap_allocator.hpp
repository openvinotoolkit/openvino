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
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define INVALID_HANDLE_VALUE -1
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
        : _path(path), _offset(offset), _size(0), _lock(lock), _data(nullptr) {
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
        closeFile();
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

    void closeFile() {
        if (_file == INVALID_HANDLE_VALUE) {
            return;
        }
#ifdef _WIN32
        ::CloseHandle(_file);
#else
        close(_file);
#endif
        _file = INVALID_HANDLE_VALUE;
    }

    void map(const char* path, size_t offset, size_t size, LockOp lock) {
#ifdef _WIN32
        SYSTEM_INFO SystemInfo;
        GetSystemInfo(&SystemInfo);
        const int64_t page_size = SystemInfo.dwAllocationGranularity;
#else
        const int64_t page_size = sysconf(_SC_PAGE_SIZE);
#endif
        const int64_t offset_align = offset / page_size * page_size;
        const int64_t map_size = offset - offset_align + size;
        size_t file_size;
#ifdef _WIN32
        DWORD file_mode;
        DWORD map_mode;
        DWORD access;

        switch (lock) {
            case LOCK_FOR_READ:
                file_mode = GENERIC_READ;
                access = PAGE_READONLY;
                map_mode = FILE_MAP_READ;
                break;
            case LOCK_FOR_WRITE:
                file_mode = GENERIC_READ | GENERIC_WRITE;
                access = PAGE_READWRITE;
                map_mode = FILE_MAP_WRITE;
                break;
            default:
                THROW_IE_EXCEPTION <<  "Unsupported lock option.";
        }

        _file = ::CreateFile(path, file_mode, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

        if (_file == INVALID_HANDLE_VALUE) {
            THROW_IE_EXCEPTION << "Can not open file for mapping.";
        }

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(handle, &file_size_large) == 0) {
            closeFile();
            THROW_IE_EXCEPTION << "Can not get file size.";
        }

        file_size = static_cast<int64_t>(file_size_large.QuadPart);

        const int64_t total_file_size = offset + size;

        if (total_file_size > file_size) {
            closeFile();
            THROW_IE_EXCEPTION << "File size is less than requested map size.";
        }

        _mapping = ::CreateFileMapping(_file, 0, access,
            total_file_size >> 32,
            total_file_size & 0xffffffff,
            0);

        if (_mapping == INVALID_HANDLE_VALUE) {
            closeFile();
            THROW_IE_EXCEPTION << "Can not create  file mapping.";
        }

        const int64_t map_size = offset - offset_align + size;

        _data = ::MapViewOfFile(
            _mapping,
            map_mode,
            offset_align >> 32,
            offset_align & 0xffffffff,
            map_size);
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

        struct stat sb = {};
        _file = open(path, mode);

        if (_file == INVALID_HANDLE_VALUE) {
            THROW_IE_EXCEPTION << "Can not open file for mapping.";
        }

        if (fstat(_file, &sb) == -1) {
            closeFile();
            THROW_IE_EXCEPTION << "Can not get file size.";
        }

        file_size = (size_t)sb.st_size;

        if (size != 0 && size > file_size) {
            closeFile();
            THROW_IE_EXCEPTION << "File size is less than requested map size.";
        }

        _size = (size == 0) ? (size_t)sb.st_size : size;

        _data = mmap(NULL, map_size, prot, MAP_PRIVATE, _file, offset_align);
    #endif
        if (_data == nullptr) {
            closeFile();
            THROW_IE_EXCEPTION << "Can not create file mapping.";
        }
    }

    void unmap() {
        if (_data != nullptr) {
#ifdef _WIN32
            ::UnmapViewOfFile(_data);
            ::CloseHandle(_mapping);
            _mapping = INVALID_HANDLE_VALUE;
#else
            munmap(_data, _size);
#endif
        }
        _data = nullptr;
        _size = 0;
    }

private:
    std::string _path;
    size_t _offset;
    size_t _size;
    LockOp _lock;

    void* _data;
#ifdef _WIN32
    HANDLE _file = INVALID_HANDLE_VALUE;
    HANDLE _mapping = INVALID_HANDLE_VALUE;
#else
    int _file = INVALID_HANDLE_VALUE;
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
