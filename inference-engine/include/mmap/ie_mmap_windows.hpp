// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file defines utility Mmap class
 * 
 * @file ie_mmap_windows.hpp
 */
#pragma once

#include "ie_immap.hpp"

#ifdef _WIN32
#include <windows.h>

namespace InferenceEngine {
/*
 * @brief This is linux implementation for memmory mapped file
 */
class MmapWindows : public IMmap {
public:
    explicit MmapWindows(const path_type& path, size_t size = 0, size_t offset = 0, LockOp lock = LOCK_FOR_READ) {
        map(path, size, offset, lock);
    }

protected:
    /**
     * @brief Unmap mapped file.
     */
    void map(const path_type& path, size_t size, size_t offset, LockOp lock) override {
        SYSTEM_INFO SystemInfo;
        GetSystemInfo(&SystemInfo);

        const int64_t page_size = SystemInfo.dwAllocationGranularity;
        const int64_t offset_align = offset / page_size * page_size;

        int64_t file_size;
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

#if defined(ENABLE_UNICODE_PATH_SUPPORT)
        _file = ::CreateFileW(path.c_str(), file_mode, FILE_SHARE_READ | FILE_SHARE_WRITE,
            0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
#else
        _file = ::CreateFileA(path.c_str(), file_mode, FILE_SHARE_READ | FILE_SHARE_WRITE,
            0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
#endif
        if (_file == INVALID_HANDLE_VALUE) {
            THROW_IE_EXCEPTION << "Can not open file for mapping.";
        }

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(_file, &file_size_large) == 0) {
            close();
            THROW_IE_EXCEPTION << "Can not get file size.";
        }

        file_size = static_cast<int64_t>(file_size_large.QuadPart);
        const int64_t map_size = offset - offset_align + (size == 0 ? file_size : size);
        const int64_t total_file_size = offset + map_size;

        if (total_file_size > file_size) {
            close();
            THROW_IE_EXCEPTION << "File size is less than requested map size.";
        }

        _mapping = ::CreateFileMapping(_file, 0, access,
            total_file_size >> 32,
            total_file_size & 0xffffffff,
            0);

        if (_mapping == INVALID_HANDLE_VALUE) {
            close();
            THROW_IE_EXCEPTION << "Can not create  file mapping.";
        }

        _data = ::MapViewOfFile(
            _mapping,
            map_mode,
            offset_align >> 32,
            offset_align & 0xffffffff,
            map_size);

        if (_data == nullptr) {
            close();
            THROW_IE_EXCEPTION << "Can not create file mapping.";
        }

        _size = map_size;
    }

    /**
     * @brief Unmap mapped file.
     */
    void unmap() override {
        if (_data != nullptr) {
            ::UnmapViewOfFile(_data);
            ::CloseHandle(_mapping);
            _mapping = INVALID_HANDLE_VALUE;
        }
        _data = nullptr;
        _size = 0;
    }

    /**
     * @brief Close file.
     */
    void close() {
        if (_file == INVALID_HANDLE_VALUE) {
            return;
        }
        ::CloseHandle(_file);
        _file = INVALID_HANDLE_VALUE;
    }

private:
    HANDLE _file = INVALID_HANDLE_VALUE;
    HANDLE _mapping = INVALID_HANDLE_VALUE;
};

/**
 * @brief Creates a memory mapped file
 * @param path Path to a file
 * @param size Size to map, 0 to entire file size
 * @param offset Offset from beggining of file to map
 * @param lock Map mode
 * @return A new Mmap
 */

std::shared_ptr<IMmap> make_mmap(const path_type& path, size_t size = 0, size_t offset = 0, LockOp lock = LOCK_FOR_READ) {
    return shared_from_irelease(new MmapWindows(path, size, offset, lock));
}
}  // namespace InferenceEngine
#endif
