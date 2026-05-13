// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared memory map objects
 * @file mmap_object.hpp
 */

#pragma once

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace ov {

namespace util {
int64_t get_system_page_size();
}  // namespace util

#ifdef _WIN32
// Windows uses HANDLE (void*) for file handles
using FileHandle = void*;
#else
// Linux/Unix uses int for file descriptors
using FileHandle = int;
#endif
/**
 * @brief Generic constant to indicate automatic size calculation is required.
 */
inline constexpr auto auto_size = std::numeric_limits<size_t>::max();

/**
 * @brief This class represents a mapped memory.
 * Instead of reading files, we can map the memory via mmap for Linux or MapViewOfFile for Windows.
 * The MappedMemory class is a abstraction to handle such memory with os-dependent details.
 */
class MappedMemory {
public:
    virtual char* data() noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual uint64_t get_id() const noexcept = 0;
    virtual ~MappedMemory() = default;
    virtual void hint_evict(size_t offset = 0, size_t size = auto_size) noexcept = 0;
};

/**
 * @brief Returns mapped memory for a file from provided path.
 * Instead of reading files, we can map the memory via mmap for Linux
 * in order to avoid time-consuming reading and reduce memory consumption.
 *
 * @param path Path to a file which memory will be mmaped.
 * @param offset Offset in the file where the mapping starts.
 * @param size Size of the mapping. If size is std::numeric_limits<size_t>::max(), maps from offset to EOF.
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path,
                                                   size_t offset = 0,
                                                   size_t size = auto_size);

/**
 * @brief Returns mapped memory for a file from provided file handle (cross-platform).
 * Uses mmap on Linux/Unix (with file descriptor) or MapViewOfFile on Windows (with HANDLE).
 * This allows external control of file access through a callback function.
 *
 * @param handle Platform-specific file handle (int fd on Linux, HANDLE on Windows).
 * @param offset Offset in the file where the mapping starts.
 * @param size Size of the mapping. If size is std::numeric_limits<size_t>::max(), maps from offset to EOF.
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle, size_t offset = 0, size_t size = auto_size);
}  // namespace ov
