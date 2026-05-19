// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>

#include <fstream>

#include "atomic_guard.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/parallel_read_streambuf.hpp"

namespace ov {
LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_reserved_size{0},
      m_reserved_buffer{nullptr},
      m_loaded{false} {
    m_byte_size = byte_size;
    const auto file_size = util::file_size(m_file_path);
    OPENVINO_ASSERT(file_size >= 0 && m_offset <= static_cast<size_t>(file_size) &&
                        m_byte_size <= static_cast<size_t>(file_size) - m_offset,
                    "If file exists its size is smaller than requested range (file size: ",
                    file_size,
                    ", requested offset: ",
                    m_offset,
                    ", requested byte size: ",
                    m_byte_size,
                    ").");

    m_reserved_size = m_byte_size + default_alignment - 1;
    OPENVINO_ASSERT(m_reserved_size >= m_byte_size,
                    "Integer overflow occurred while calculating reserved size for LazyBuffer (requested byte size: ",
                    m_byte_size,
                    ", alignment: ",
                    default_alignment,
                    ").");
    m_reserved_buffer = VirtualAlloc(nullptr, m_reserved_size, MEM_RESERVE, PAGE_NOACCESS);
    OPENVINO_ASSERT(m_reserved_buffer != nullptr, "VirtualAlloc reserve failed, err: ", GetLastError());

    m_aligned_buffer = static_cast<char*>(m_reserved_buffer) +
                       util::align_padding_size(default_alignment, reinterpret_cast<size_t>(m_reserved_buffer));
}

LazyBuffer::~LazyBuffer() {
    if (m_reserved_buffer != nullptr) {
        std::ignore = VirtualFree(m_reserved_buffer, 0, MEM_RELEASE);
    }
}

void LazyBuffer::hint_prefetch() const {
    std::lock_guard lock{m_loading};
    if (!m_loaded && m_byte_size > 0) {
        if (!VirtualAlloc(static_cast<char*>(m_reserved_buffer), m_reserved_size, MEM_COMMIT, PAGE_READWRITE)) {
            OPENVINO_THROW("VirtualAlloc commit failed, err: ", GetLastError());
        }

        try {
            util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
            std::istream file(&par_buf);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
            file.read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
        } catch (...) {
            std::ignore = VirtualFree(m_reserved_buffer, m_reserved_size, MEM_DECOMMIT);
            throw;
        }
        m_loaded = true;
    }
}

void LazyBuffer::hint_evict() noexcept {
    std::lock_guard lock{m_loading};
    if (m_loaded) {
        m_loaded = false;
        std::ignore = VirtualFree(m_reserved_buffer, m_reserved_size, MEM_DECOMMIT);
    }
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (offset == 0 && size >= m_byte_size) {
        hint_evict();
    }
}
}  // namespace ov
