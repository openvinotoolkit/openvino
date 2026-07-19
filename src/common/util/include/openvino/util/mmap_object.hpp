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

#include "openvino/util/file_util.hpp"

namespace ov {

namespace util {
int64_t get_system_page_size();
}  // namespace util

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
    /**
     * @brief Hint that the given region of the mapping will be accessed soon.
     *
     * @param offset Offset within the mapping where prefetching starts.
     * @param size   Number of bytes to prefetch. Defaults to the rest of the
     *               mapping when set to auto_size.
     */
    virtual void hint_prefetch(size_t offset = 0, size_t size = auto_size) = 0;
};

/**
 * @brief Returns mapped memory for a file from provided path.
 * Instead of reading files, we can map the memory via mmap for Linux
 * in order to avoid time-consuming reading and reduce memory consumption.
 *
 * @param path Path to a file which memory will be mmaped.
 * @param offset Offset in the file where the mapping starts.
 * @param size Size of the mapping. If size is std::numeric_limits<size_t>::max(), maps from offset to EOF.
 * @param no_placeholder When true, skip the Windows 10+ placeholder/VEH mechanism and use the legacy
 *                       single-call MapViewOfFile path instead. This guarantees a uniform AllocationBase
 *                       across the whole mapping, required for NPU zero-copy blob import. On Linux ignored.
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path,
                                                   size_t offset = 0,
                                                   size_t size = auto_size,
                                                   bool no_placeholder = false);

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
