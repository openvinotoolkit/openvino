// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of platform-specific parallel I/O primitives.
 * @file parallel_io.hpp
 */

#pragma once

#include <cstddef>
#include <filesystem>

namespace ov::util {

#ifdef _WIN32
/// Platform file handle: Windows HANDLE (void*).
using FileHandle = void*;
inline const FileHandle INVALID_FILE_HANDLE = reinterpret_cast<void*>(-1);  // NOLINT(performance-no-int-to-ptr)
#else
/// Platform file handle: Linux/Unix file descriptor (int).
using FileHandle = int;
inline constexpr FileHandle INVALID_FILE_HANDLE = -1;
#endif

inline constexpr size_t DEFAULT_PARALLEL_IO_THRESHOLD = 4UL * 1024 * 1024;  ///< 4 MB default threshold for parallel I/O
inline constexpr size_t DEFAULT_PARALLEL_IO_MIN_CHUNK = 2UL * 1024 * 1024;  ///< 2 MB minimum chunk size per thread

/**
 * @brief Open a file for reading and retrieve its size.
 *
 * On Linux, uses open(O_RDONLY | O_CLOEXEC) + fstat.
 * On Windows, uses CreateFileW(GENERIC_READ) + GetFileSizeEx.
 *
 * @param path          Path to the file to open.
 * @param file_offset   Header offset to validate (must be within [0, file_size]).
 * @param out_handle    [out] The opened file handle / descriptor.
 * @param out_size      [out] Total size of the file in bytes.
 * @throws std::runtime_error  If the file cannot be opened or its size cannot be queried.
 * @throws std::out_of_range   If file_offset is outside [0, file_size].
 */
void get_file_handle_and_size(const std::filesystem::path& path,
                              std::streamoff file_offset,
                              FileHandle& out_handle,
                              std::streamoff& out_size);

/**
 * @brief Close a platform file handle.
 *
 * On Linux, calls close(fd).  On Windows, calls CloseHandle(handle).
 * Safe to call with an invalid handle (no-op).
 *
 * @param handle  The file handle to close.
 */
void close_file_handle(FileHandle handle);

/**
 * @brief Open a file for reading (lightweight, for per-thread file handles).
 *
 * Each worker thread in a parallel read should open its own file handle so that
 * the OS readahead state is independent per thread.
 *
 * On Linux, uses open(O_RDONLY | O_CLOEXEC).
 * On Windows, uses CreateFileW(GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE).
 *
 * @param path  Path to the file.
 * @return A valid file handle, or the platform-specific invalid value on failure.
 */
FileHandle open_file_for_read(const std::filesystem::path& path);

/**
 * @brief Read bytes from a file at a given absolute offset (single-threaded).
 *
 * On Linux, uses pread() in a loop.
 * On Windows, uses SetFilePointerEx + ReadFile in a loop.
 *
 * @note This function is @b not thread-safe for a given handle.  Callers that
 *       perform parallel reads must open a separate handle per thread (see
 *       open_file_for_read).
 *
 * @param handle       File handle / descriptor.
 * @param dst          Destination buffer.
 * @param size         Number of bytes to read.
 * @param file_offset  Absolute byte offset in the file.
 * @return true if all bytes were read successfully, false on I/O error.
 */
bool positional_read(FileHandle handle, char* dst, size_t size, size_t file_offset);

/**
 * @brief Detect whether a memory address is backed by a file-based mmap.
 *
 * On Linux, parses /proc/self/maps.
 * On Windows, uses VirtualQuery + GetMappedFileNameW + drive letter resolution.
 *
 * @param addr        The memory address to inspect.
 * @param out_path    [out] Path to the backing file (only set on success).
 * @param out_offset  [out] Absolute byte offset within the file corresponding to addr.
 * @return true if the address is file-backed, false otherwise.
 */
bool get_mmap_file_info(const void* addr, std::filesystem::path& out_path, std::streamoff& out_offset);

/**
 * @brief Issue an asynchronous prefetch hint for a memory region.
 *
 * On Linux, calls madvise(MADV_WILLNEED) (with page-aligned address).
 * On Windows, calls PrefetchVirtualMemory.
 *
 * @param addr  Start of the memory region.
 * @param size  Size of the region in bytes.
 */
void prefetch_memory(const void* addr, size_t size);

}  // namespace ov::util