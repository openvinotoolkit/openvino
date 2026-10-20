// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Platform-agnostic I/O utilities.
 * @file io.hpp
 */

#pragma once

#include <cstddef>
#include <system_error>

#include "openvino/util/file_util.hpp"

namespace ov::util {

/**
 * @brief Populate a mapped memory region using the native async I/O backend (or synchronous fallback).
 *
 * Splits the target range into at most @p queue_depth chunks and populates them in parallel.
 * When the native async I/O ring is unavailable, a synchronous OS memory hint is issued instead.
 *
 * @param ptr         Base address of the mapped region (as returned by the OS mapping call or @ref load_mmap_object).
 * @param size        Number of bytes to populate.
 * @param offset      Byte offset within the mapping to start from. Default: 0.
 * @param queue_depth Maximum number of parallel population operations. Default: 128.
 * @return @c true on success or after issuing the advisory hint.
 *         @c false only if @p ptr is @c nullptr or @p size is @c 0.
 *
 * @code
 * auto mm = ov::util::load_mmap_object("model.bin");
 *
 * // Populate the entire mapping:
 * ov::util::io_populate_mmap(mm->data(), mm->size());
 * @endcode
 */
bool io_populate_mmap(void* ptr, size_t size, size_t offset = 0, size_t queue_depth = 128) noexcept;

/**
 * @brief Read a region from an open file directly into any destination buffer.
 *
 * Transfers [@p file_offset, @p file_offset + @p size) from @p handle into @p dst using the native async I/O backend
 *  or the thread-pool fallback.
 *
 * The caller retains ownership of @p handle; it is not closed by this function.
 * If data is already in memory, use @c memcpy instead.
 *
 * The destination can be any writable buffer: heap memory, device memory, staging buffers, etc. No memory-mapped pages
 * are touched, so the resident-set size of any existing mapping of the same file is unchanged.
 *
 * @param handle      Open, readable platform file handle.
 * @param dst         Destination buffer.  Must hold at least @p size bytes.
 * @param file_offset Byte offset within the file to start reading.
 * @param size        Number of bytes to read.
 * @param queue_depth Submission-queue depth for the async I/O backend (ignored on the fallback path).
 * @return An empty @c std::error_code on success; a non-zero code describing the failure otherwise.
 */
std::error_code io_read_into(FileHandle handle,
                             void* dst,
                             size_t file_offset,
                             size_t size,
                             size_t queue_depth = 128) noexcept;
}  // namespace ov::util
