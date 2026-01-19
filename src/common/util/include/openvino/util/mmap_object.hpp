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
#include <memory>
#include <string>

namespace ov {

#ifdef _WIN32
// Windows uses HANDLE (void*) for file handles
using FileHandle = void*;
#else
// Linux/Unix uses int for file descriptors
using FileHandle = int;
#endif

/**
 * @brief This class represents a mapped memory.
 * Instead of reading files, we can map the memory via mmap for Linux or MapViewOfFile for Windows.
 * The MappedMemory class is a abstraction to handle such memory with os-dependent details.
 */
class MappedMemory {
public:
    virtual char* data() noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual ~MappedMemory() = default;
};

/**
 * @brief Returns mapped memory for a file from provided path.
 * Instead of reading files, we can map the memory via mmap for Linux
 * in order to avoid time-consuming reading and reduce memory consumption.
 *
 * @param path Path to a file which memory will be mmaped.
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path);

/**
 * @brief Returns mapped memory for a file from provided file handle (cross-platform).
 * Uses mmap on Linux/Unix (with file descriptor) or MapViewOfFile on Windows (with HANDLE).
 * This allows external control of file access through a callback function.
 *
 * @param handle Platform-specific file handle (int fd on Linux, HANDLE on Windows).
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle);
}  // namespace ov
