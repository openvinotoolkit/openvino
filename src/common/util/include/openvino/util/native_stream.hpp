// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief An ifstream-like input stream backed by NativeStreamBuf (io_read_into platform backend).
 * @file native_stream.hpp
 */

#pragma once

#include <filesystem>
#include <istream>

#include "openvino/util/native_streambuf.hpp"

namespace ov::util {

/**
 * @brief A read-only std::istream backed by NativeStreamBuf.
 *
 * Uses native OS file handles and the platform io_read_into backend instead of C-runtime stdio,
 * making it a drop-in replacement for std::ifstream for binary sequential and random-access reads.
 *
 * Three construction modes:
 *  - **default constructor**: stream not associated with any file; behaves like a default-constructed
 *    std::ifstream — reads immediately return EOF.
 *  - **path constructor** (owning): opens a read-only native handle from a path; closes it on destruction.
 *  - **handle constructor** (non-owning): borrows a caller-provided handle for a sub-region
 *    [@p offset, @p offset + @p size); the handle is never closed by this class.
 *
 * Supports move construction and move assignment (via swap). Copy is deleted.
 *
 * @code
 *   ov::util::NativeIfstream empty;                      // default — not associated with any file
 *   ov::util::NativeIfstream file(path);                 // owning — whole file from offset 0
 *   ov::util::NativeIfstream view(handle, off, size);    // non-owning — sub-region [off, off+size)
 *   file.seekg(blob_offset);
 *   file.read(dst, n);
 * @endcode
 */
class NativeIfstream : public std::istream {
public:
    /** @brief Constructs a stream not associated with any file. Any read immediately returns EOF. */
    NativeIfstream() noexcept;

    /**
     * @brief Opens a read-only native handle from @p path and takes ownership of it.
     *
     * The stream is positioned at the beginning of the file. Use @c seekg() to reposition.
     *
     * @param path  Path to the file to open.
     */
    explicit NativeIfstream(const std::filesystem::path& path);

    /**
     * @brief Builds a stream over a caller-owned native handle. The handle is never closed by this class.
     *
     * The caller must keep @p handle valid for the entire lifetime of the stream.
     *
     * @param handle  Read-only native handle. Ownership remains with the caller.
     * @param offset  Absolute file offset that maps to logical stream position 0.
     * @param size    Number of readable bytes from @p offset.
     */
    explicit NativeIfstream(FileHandle handle, std::streamoff offset, std::streamoff size);

    ~NativeIfstream();

    NativeIfstream(const NativeIfstream&) = delete;
    NativeIfstream& operator=(const NativeIfstream&) = delete;

    NativeIfstream(NativeIfstream&& other) noexcept;
    NativeIfstream& operator=(NativeIfstream&& other) noexcept;

    void swap(NativeIfstream& other) noexcept;

private:
    FileHandle m_handle;    //!< Native file handle; may be owned or borrowed.
    bool m_owns_handle;     //!< True when the destructor is responsible for closing m_handle.
    NativeStreamBuf m_buf;  //!< Streambuf value member; Must be after m_handle and m_owns_handle.
};

}  // namespace ov::util
