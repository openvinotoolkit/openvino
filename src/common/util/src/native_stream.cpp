// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/native_stream.hpp"

#include <utility>

#include "openvino/util/file_util.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

// Default constructor: stream not associated with any file; reads immediately return EOF.
NativeIfstream::NativeIfstream() noexcept
    : std::istream(nullptr),
      m_handle(INVALID_HANDLE_VALUE),
      m_owns_handle(false),
      m_buf() {
    this->init(&m_buf);
}

NativeIfstream::NativeIfstream(const std::filesystem::path& path)
    : std::istream(nullptr),
      m_handle(open_file(path, FileMode::read | FileMode::direct)),
      m_owns_handle(m_handle != INVALID_HANDLE_VALUE),
      m_buf(m_handle, 0, m_owns_handle ? static_cast<std::streamoff>(ov::util::file_size(path)) : 0) {
    this->init(&m_buf);
    if (!m_owns_handle)
        setstate(std::ios::failbit);
}

NativeIfstream::NativeIfstream(FileHandle handle, std::streamoff offset, std::streamoff size)
    : std::istream(nullptr),
      m_handle(handle),
      m_owns_handle(false),
      m_buf(handle, offset, size) {
    this->init(&m_buf);
}

NativeIfstream::NativeIfstream(NativeIfstream&& other) noexcept
    : std::istream(std::move(other)),
      m_handle(std::exchange(other.m_handle, INVALID_HANDLE_VALUE)),
      m_owns_handle(std::exchange(other.m_owns_handle, false)),
      m_buf(std::move(other.m_buf)) {
    this->set_rdbuf(&m_buf);
}

NativeIfstream& NativeIfstream::operator=(NativeIfstream&& other) noexcept {
    if (this != &other) {
        swap(other);  // old *this state (handle, buf) migrates to other; closed by other's destructor
    }
    return *this;
}

void NativeIfstream::swap(NativeIfstream& other) noexcept {
    std::istream::swap(other);
    std::swap(m_handle, other.m_handle);
    std::swap(m_owns_handle, other.m_owns_handle);
    m_buf.swap(other.m_buf);
}

NativeIfstream::~NativeIfstream() {
    if (m_owns_handle) {
        close_file_handle(m_handle);
    }
}

}  // namespace ov::util
