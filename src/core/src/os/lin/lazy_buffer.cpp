// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
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
      m_reserved_buffer{MAP_FAILED},
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
    m_reserved_buffer = mmap(nullptr, m_reserved_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    OPENVINO_ASSERT(m_reserved_buffer != MAP_FAILED, "mmap failed, err: ", std::strerror(errno));

    m_aligned_buffer = static_cast<char*>(m_reserved_buffer) +
                       util::align_padding_size(default_alignment, reinterpret_cast<size_t>(m_reserved_buffer));
}

LazyBuffer::~LazyBuffer() {
    if (m_reserved_buffer != MAP_FAILED) {
        std::ignore = munmap(m_reserved_buffer, m_reserved_size);
    }
}

void LazyBuffer::hint_prefetch() const {
    std::lock_guard lock{m_loading};
    if (!m_loaded && m_byte_size > 0) {
        if (mprotect(m_reserved_buffer, m_reserved_size, PROT_READ | PROT_WRITE) == -1) {
            OPENVINO_THROW("mprotect failed, err: ", std::strerror(errno));
        }
        try {
            util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
            std::istream file(&par_buf);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
            file.read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
        } catch (...) {
            std::ignore = mprotect(m_reserved_buffer, m_reserved_size, PROT_NONE);
            throw;
        }
        m_loaded = true;
    }
}

void LazyBuffer::hint_evict() noexcept {
    std::lock_guard lock{m_loading};
    if (m_loaded) {
        m_loaded = false;
        std::ignore = mprotect(m_reserved_buffer, m_reserved_size, PROT_NONE);
    }
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (offset == 0 && size >= m_byte_size) {
        hint_evict();
    }
}
}  // namespace ov
