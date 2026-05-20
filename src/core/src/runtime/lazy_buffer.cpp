// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <istream>
#include <mutex>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/parallel_read_streambuf.hpp"
#include "openvino/util/reservable_buffer.hpp"

namespace ov {
LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_loaded{false} {
    OPENVINO_ASSERT(byte_size > 0, "Zero size buffer makes no sense.");
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
    const auto reserve_size = m_byte_size + default_alignment - 1;
    OPENVINO_ASSERT(reserve_size >= m_byte_size,
                    "Integer overflow occurred while calculating reserved size for LazyBuffer (requested byte size: ",
                    m_byte_size,
                    ", alignment: ",
                    default_alignment,
                    ").");

    m_impl = std::make_unique<util::ReservableBuffer>(reserve_size);
    const auto reserved_buffer = m_impl->reserve();
    OPENVINO_ASSERT(reserved_buffer != nullptr,
                    "Failed to reserve memory for LazyBuffer (requested byte size: ",
                    m_byte_size,
                    ", reserved size: ",
                    reserve_size,
                    "). Error: ",
                    m_impl->get_last_error());
    m_aligned_buffer = static_cast<char*>(reserved_buffer) +
                       util::align_padding_size(default_alignment, reinterpret_cast<size_t>(reserved_buffer));
}

LazyBuffer::LazyBuffer(LazyBuffer&& other) noexcept
    : AlignedBuffer(std::move(other)),
      m_file_path{std::move(other.m_file_path)},
      m_offset{other.m_offset},
      m_loaded{other.m_loaded.load()},
      m_impl{std::move(other.m_impl)} {
    other.m_loaded = false;
    other.m_offset = 0;
};

LazyBuffer& LazyBuffer::operator=(LazyBuffer&& other) noexcept {
    if (this != &other) {
        AlignedBuffer::operator=(std::move(other));
        m_file_path = std::move(other.m_file_path);
        m_offset = other.m_offset;
        m_loaded = other.m_loaded.load();
        m_impl = std::move(other.m_impl);

        other.m_loaded = false;
        other.m_offset = 0;
    }
    return *this;
}

LazyBuffer::~LazyBuffer() = default;

void LazyBuffer::hint_prefetch() const {
    if (!m_loaded) {
        std::lock_guard lock{m_loading};
        if (m_loaded) {
            return;
        }
        if (!m_impl->acquire()) {
            OPENVINO_THROW("Failed to acquire memory for LazyBuffer. Error: ", m_impl->get_last_error());
        }
        try {
            util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
            std::istream file(&par_buf);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
            file.read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
            m_loaded = true;
        } catch (...) {
            m_impl->evict();
            throw;
        }
    }
}

void LazyBuffer::hint_evict() noexcept {
    if (m_loaded) {
        try {
            std::lock_guard lock{m_loading};
            if (m_loaded) {
                m_impl->evict();
                m_loaded = false;
            }
        } catch (...) {
        }
    }
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (offset == 0 && size >= m_byte_size) {
        hint_evict();
    }
}
}  // namespace ov
