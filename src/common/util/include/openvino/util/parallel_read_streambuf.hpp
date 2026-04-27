// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of a parallel I/O streambuf for file-based reads.
 * @file parallel_read_streambuf.hpp
 */

#pragma once

#include <filesystem>
#include <memory>
#include <streambuf>

#include "openvino/util/parallel_io.hpp"

namespace ov::util {

/**
 * @brief A std::streambuf that reads from a file using parallel I/O for large
 *        reads, bypassing the OS page cache pressure that mmap+memcpy incurs.
 *
 * For reads >= threshold bytes, the read is split across N threads where each
 * thread issues its own independent positional read operation using
 * platform-specific APIs. Smaller reads fall through to a single positional call.
 *
 * Usage:
 * @code
 *   ParallelReadStreamBuf buf(cache_path, blob_offset_in_file);
 *   std::istream stream(&buf);
 *   cldnn::BinaryInputBuffer ib(stream, engine);
 *   ib >> ...;
 * @endcode
 */
class ParallelReadStreamBuf : public std::streambuf {
public:
    /**
     * @param path           Path to the file to read.
     * @param header_offset  Initial file position (absolute offset from the start
     *                       of the file; the stream starts reading from here).
     * @param threshold      Minimum read size to trigger parallel I/O.
     */
    explicit ParallelReadStreamBuf(const std::filesystem::path& path,
                                   std::streamoff header_offset = 0,
                                   size_t threshold = default_parallel_io_threshold);

    ~ParallelReadStreamBuf() override;

    ParallelReadStreamBuf(const ParallelReadStreamBuf&) = delete;
    ParallelReadStreamBuf& operator=(const ParallelReadStreamBuf&) = delete;

    /**
     * @brief Preload @p size bytes starting at the current logical position into
     *        an internal buffer using one parallel positional read.
     *
     * After a successful prefetch, subsequent xsgetn()/underflow() calls that
     * fall inside [current_pos, current_pos + size) are served from memory via
     * memcpy instead of issuing per-call pread(). Reads that fall outside the
     * prefetched window transparently fall back to the normal file-IO path and
     * invalidate the prefetched window.
     *
     * Intended call site: the producer of a long serialized region (e.g.
     * program::weights_load in the GPU plugin) calls prefetch() once at the
     * start of the region to collapse thousands of small ib >> ... small-reads
     * into a single bulk parallel pread.
     *
     * @param size  Number of bytes to preload. Clamped to remaining file size.
     * @return true if the prefetch read succeeded (buffer is now valid), false
     *         otherwise (buffer is left empty; reads fall back to file I/O).
     */
    bool prefetch(std::streamsize size);

protected:
    std::streamsize xsgetn(char_type* dst, std::streamsize n) override;
    int_type underflow() override;
    pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
    std::streamsize showmanyc() override;

private:
    bool single_read(char* dst, size_t size, size_t file_offset);
    bool parallel_read(char* dst, size_t size, size_t file_offset);

    static constexpr size_t UNDERFLOW_BUF = 8192;  ///< batch size for char-by-char reads

    std::filesystem::path m_path;
    FileHandle m_handle;  ///< platform file handle

    std::streamoff m_file_offset = 0;    ///< absolute file offset of next byte to read
    std::streamoff m_header_offset = 0;  ///< absolute file offset of logical stream start
    std::streamoff m_file_size = 0;
    size_t m_threshold = default_parallel_io_threshold;
    std::unique_ptr<char_type[]> m_underflow_buf;  ///< lazily allocated buffer for underflow()

    /// Prefetch buffer state. When m_prefetch_size > 0, reads in the range
    /// [m_prefetch_begin, m_prefetch_begin + m_prefetch_size) are served from
    /// m_prefetch_buf without touching the file.
    std::unique_ptr<char_type[]> m_prefetch_buf;
    std::streamoff m_prefetch_begin = 0;  ///< absolute file offset of prefetch buffer[0]
    size_t m_prefetch_size = 0;           ///< bytes loaded into m_prefetch_buf

    bool serve_from_prefetch(char* dst, size_t size, std::streamoff abs_offset);
    void invalidate_prefetch();
};

}  // namespace ov::util