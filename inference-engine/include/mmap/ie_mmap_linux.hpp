// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file defines utility Mmap class
 * 
 * @file ie_mmap.hpp
 */
#pragma once

#include "ie_immap.hpp"

#if defined(LINUX) || defined(__APPLE__)

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define INVALID_HANDLE_VALUE -1

namespace InferenceEngine {
/*
 * @brief This is linux implementation for memmory mapped file
 */
class MmapLinux : public IMmap {
public:
    explicit MmapLinux(const path_type& path, size_t size = 0, size_t offset = 0, LockOp lock = LOCK_FOR_READ) {
        map(path, size, offset, lock);
    }

protected:
    /**
     * @brief Unmap mapped file.
     */
    void map(const path_type& path, size_t size, size_t offset, LockOp lock) override {
        const int64_t page_size = sysconf(_SC_PAGE_SIZE);
        int64_t file_size;

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
        _file = open(path.c_str(), mode);

        if (_file == INVALID_HANDLE_VALUE) {
            THROW_IE_EXCEPTION << "Can not open file for mapping.";
        }

        if (fstat(_file, &sb) == -1) {
            close();
            THROW_IE_EXCEPTION << "Can not get file size.";
        }

        file_size = (size_t)sb.st_size;

        const int64_t offset_align = offset / page_size * page_size;
        const int64_t map_size = offset - offset_align + (size == 0 ? file_size : size);
        const int64_t total_file_size = offset + map_size;

        if (total_file_size > file_size) {
            close();
            THROW_IE_EXCEPTION << "File size is less than requested map size.";
        }

        _data = mmap(NULL, map_size, prot, MAP_PRIVATE, _file, offset_align);

        if (_data == MAP_FAILED) {
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
            munmap(_data, _size);
        }
        _data = nullptr;
        _size = 0;
    }

private:
    /**
     * @brief Close file.
     */
    void close() {
        if (_file == INVALID_HANDLE_VALUE) {
            return;
        }
        ::close(_file);
        _file = INVALID_HANDLE_VALUE;
    }

private:
    int _file = INVALID_HANDLE_VALUE;
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
    return shared_from_irelease(new MmapLinux(path, size, offset, lock));
}
}  // namespace InferenceEngine

#endif
