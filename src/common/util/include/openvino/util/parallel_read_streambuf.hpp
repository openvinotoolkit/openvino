// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of a parallel I/O streambuf for file-based reads.
 * @file parallel_read_streambuf.hpp
 */

#pragma once

#include <array>
#include <filesystem>
#include <streambuf>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    ifdef min
#        undef min
#    endif
#    ifdef max
#        undef max
#    endif
#endif

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
    static constexpr size_t DEFAULT_THRESHOLD = 4UL * 1024 * 1024;  ///< 4 MB

    /**
     * @param path           Path to the file to read.
     * @param header_offset  Initial file position (absolute offset from the start
     *                       of the file; the stream starts reading from here).
     * @param threshold      Minimum read size to trigger parallel I/O.
     */
    explicit ParallelReadStreamBuf(const std::filesystem::path& path,
                                   std::streamoff header_offset = 0,
                                   size_t threshold = DEFAULT_THRESHOLD);

    ~ParallelReadStreamBuf() override;

    ParallelReadStreamBuf(const ParallelReadStreamBuf&) = delete;
    ParallelReadStreamBuf& operator=(const ParallelReadStreamBuf&) = delete;

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
#ifdef _WIN32
    HANDLE m_handle = INVALID_HANDLE_VALUE;  ///< Windows file HANDLE
#else
    int m_fd = -1;  ///< Linux file descriptor
#endif

    std::streamoff m_file_offset = 0;        ///< absolute file offset of next byte to read
    std::streamoff m_header_offset = 0;  ///< absolute file offset of logical stream start
    std::streamoff m_file_size = 0;
    size_t m_threshold = DEFAULT_THRESHOLD;
    std::array<char_type, UNDERFLOW_BUF> m_underflow_buf{};  ///< buffer for underflow()
};

}  // namespace ov::util