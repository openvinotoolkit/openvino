// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for a handle-based fast-read streambuf.
 * @file native_streambuf.hpp
 */

#pragma once

#include <cstddef>
#include <streambuf>

#include "openvino/util/file_util.hpp"

namespace ov::util {

/** @brief Default size of the internal window used to amortize small sequential reads. */
inline constexpr size_t default_native_window = 8UL * 1024 * 1024;

/** @brief Reads at least this large bypass the window and are read straight into the caller buffer. */
inline constexpr size_t default_native_threshold = 2UL * 1024 * 1024;

/**
 * @brief A std::streambuf that reads from an already-open file handle using the platform-agnostic I/O backend.
 *
 * The buffer is decoupled from any file path: it only holds a (non-owning) file handle and serves reads through
 * @ref io_read_into. Bulk reads (>= threshold) are passed straight to the backend into the caller's destination
 * buffer, saturating fast storage without an intermediate copy. Smaller, char-by-char reads are amortized through
 * a single internal window buffer.
 *
 * The handle is @b not owned by this object; the caller is responsible for keeping it open for the whole lifetime
 * of the streambuf and for closing it afterwards.
 *
 * Usage:
 * @code
 *   FileHandle handle = ...;        // opened and sized by the caller
 *   std::streamoff size = ...;      // readable bytes from blob_offset to the end of the region
 *   NativeStreamBuf buf(handle, blob_offset, size);
 *   std::istream stream(&buf);
 *   ib >> ...;
 * @endcode
 */
class NativeStreamBuf : public std::streambuf {
public:
    /** @brief Constructs an empty streambuf not associated with any file. Any read immediately returns EOF. */
    NativeStreamBuf() noexcept;

    /**
     * @param handle        Open, readable file handle. Not owned; must outlive this object.
     *                      Pass @c INVALID_HANDLE_VALUE only via the default constructor.
     * @param offset        Absolute file offset mapped to logical stream position 0. Must be >= 0.
     * @param size          Number of readable bytes starting at @p offset. Must be >= 0;
     *                      pass 0 to create an empty (EOF-only) region. Caller is responsible for
     *                      ensuring the range [@p offset, @p offset + @p size) lies
     *                      within the file; out-of-range reads fail at the io_read_into level.
     * @param window        Size of the internal amortization window in bytes.
     * @param threshold     Minimum read size that bypasses the window and reads straight into the destination.
     */
    explicit NativeStreamBuf(FileHandle handle,
                             std::streamoff offset,
                             std::streamoff size,
                             size_t window = default_native_window,
                             size_t threshold = default_native_threshold);

    ~NativeStreamBuf() override;

    NativeStreamBuf(const NativeStreamBuf&) = delete;
    NativeStreamBuf& operator=(const NativeStreamBuf&) = delete;

    NativeStreamBuf(NativeStreamBuf&& other) noexcept;
    NativeStreamBuf& operator=(NativeStreamBuf&& other) noexcept;
    void swap(NativeStreamBuf& other) noexcept;

protected:
    std::streamsize xsgetn(char_type* dst, std::streamsize n) override;
    int_type underflow() override;
    pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
    std::streamsize showmanyc() override;

private:
    bool allocate_window();
    bool read_into(char* dst, size_t size, std::streamoff abs);
    bool fill_window();

    FileHandle m_handle;          //!< Non-owning file handle; must remain open for the lifetime of this object.
    std::streamoff m_cursor = 0;  //!< Absolute offset of the next byte to be fetched.
    std::streamoff m_start = 0;   //!< Absolute offset of the region start; maps to logical stream position 0.
    std::streamoff m_end = 0;     //!< Absolute offset one past the last readable byte of the region.

    size_t m_window_capacity = 0;                     //!< allocated capacity of m_window in bytes
    size_t m_bypass_size = default_native_threshold;  //!< minimum read size that bypasses the window
    char* m_window = nullptr;                         //!< window buffer backing the get-area; null until first use
};

}  // namespace ov::util
