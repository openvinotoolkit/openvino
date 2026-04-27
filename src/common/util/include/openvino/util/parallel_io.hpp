// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of platform-specific parallel I/O primitives.
 * @file parallel_io.hpp
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <filesystem>

#include "openvino/util/mmap_object.hpp"

namespace ov::util {

#ifndef _WIN32
inline constexpr FileHandle INVALID_HANDLE_VALUE = -1;
#endif

// Threshold below which parallel I/O's per-call dispatch cost (thread creation,
// atomic coordination, per-thread fd open) dominates the serial read time.
// Above this size, splitting the read across N threads amortises the dispatch
// cost. 16 MB sits on a plateau with 32 MB; smaller values spend more time in
// dispatch than in the read itself, larger values leave I/O bandwidth on the
// table by waiting for the slowest thread.
inline constexpr size_t default_parallel_io_threshold = 16UL * 1024 * 1024;
// Minimum bytes per worker thread; caps num_threads = size / min_chunk so that
// sub-min_chunk reads stay single-threaded.
inline constexpr size_t default_parallel_io_min_chunk = 2UL * 1024 * 1024;

// Runtime overrides for field diagnosis and re-tuning on new hardware without
// a rebuild. Set OV_PARALLEL_IO_THRESHOLD_MB or OV_PARALLEL_IO_MIN_CHUNK_MB in
// the environment; unset env falls back to the constants above.
inline size_t resolve_parallel_io_threshold() {
    if (const char* s = std::getenv("OV_PARALLEL_IO_THRESHOLD_MB")) {
        if (const long v = std::atol(s); v > 0) {
            return static_cast<size_t>(v) * 1024UL * 1024UL;
        }
    }
    return default_parallel_io_threshold;
}

inline size_t resolve_parallel_io_min_chunk() {
    if (const char* s = std::getenv("OV_PARALLEL_IO_MIN_CHUNK_MB")) {
        if (const long v = std::atol(s); v > 0) {
            return static_cast<size_t>(v) * 1024UL * 1024UL;
        }
    }
    return default_parallel_io_min_chunk;
}
///< Default upper bound for ParallelReadStreamBuf::prefetch() requests.  32 MiB was
///< selected via PTLH cap-tuning sweep (4, 8, 16, 32, 64, 128, 256, 512, 1024 MiB)
///< on Qwen3-VL-4B INT4 (cache-hit): p50 is flat across 16-64 MiB (2.09-2.10 s) and
///< rises steadily for >=128 MiB as prefetch dispatch/allocation dominates.  32 MiB
///< is the smallest cap that still covers the largest sub-program's node_post_load
///< burst in a single window, leaving headroom for models larger than Qwen3-VL-4B
///< (e.g. 7-8B class LLMs) without re-tuning.  Caps <=8 MiB exhaust the window
///< mid-loop and trigger a fallback pread burst, erasing the amortisation gain.
inline constexpr size_t default_parallel_io_prefetch_cap = 32UL * 1024 * 1024;

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
}  // namespace ov::util