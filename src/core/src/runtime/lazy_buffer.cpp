// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <istream>
#include <mutex>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/parallel_read_streambuf.hpp"

namespace ov {
LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_loaded{false} {
    m_byte_size = byte_size;

    const auto file_size = util::file_size(m_file_path);
    OPENVINO_ASSERT(file_size >= 0, "Failed to get file size for ", m_file_path);

    const bool offset_fits = m_offset <= static_cast<size_t>(file_size);
    const bool size_fits = m_byte_size <= static_cast<size_t>(file_size) - m_offset;
    OPENVINO_ASSERT(offset_fits && size_fits,
                    "Requested region ",
                    m_offset,
                    "+",
                    m_byte_size,
                    " exceeds file size ",
                    file_size,
                    " for file: ",
                    m_file_path);

    m_aligned_buffer = static_cast<char*>(util::reserve_buffer(m_byte_size));
    OPENVINO_ASSERT(m_aligned_buffer != nullptr, "Failed to reserve memory for LazyBuffer");
}

LazyBuffer::~LazyBuffer() {
    util::release_buffer(m_aligned_buffer, m_byte_size);
    m_aligned_buffer = nullptr;
    m_byte_size = 0;
}

LazyBuffer::LazyBuffer(LazyBuffer&& other) noexcept
    : AlignedBuffer(std::move(other)),
      m_file_path{std::move(other.m_file_path)},
      m_offset{std::exchange(other.m_offset, 0)},
      m_loaded{other.m_loaded.exchange(false, std::memory_order_relaxed)} {}

LazyBuffer& LazyBuffer::operator=(LazyBuffer&& other) noexcept {
    if (this != &other) {
        AlignedBuffer::operator=(std::move(other));
        m_file_path = std::move(other.m_file_path);
        m_offset = std::exchange(other.m_offset, 0);
        m_loaded = other.m_loaded.exchange(false, std::memory_order_relaxed);
    }
    return *this;
}

void LazyBuffer::hint_prefetch() const {
    if (!m_loaded.load(std::memory_order_acquire)) {
        std::lock_guard lock{m_loading};
        if (m_loaded.load(std::memory_order_relaxed)) {
            return;
        }

        std::string error;
        util::acquire_buffer(m_aligned_buffer, m_byte_size, &error);
        if (!error.empty()) {
            OPENVINO_THROW("Failed to acquire memory for LazyBuffer. Error: ", error);
        }

        try {
            util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
            std::istream file(&par_buf);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
            file.read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
            m_loaded.store(true, std::memory_order_release);
        } catch (...) {
            util::evict_buffer(m_aligned_buffer, m_byte_size);
            throw;
        }
    }
}

void LazyBuffer::hint_evict() noexcept {
    hint_evict(0, m_byte_size);
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (m_loaded.load(std::memory_order_acquire)) {
        try {
            std::lock_guard lock{m_loading};
            if (m_loaded.load(std::memory_order_relaxed)) {
                util::evict_buffer(m_aligned_buffer, m_byte_size);
                m_loaded.store(false, std::memory_order_release);
            }
        } catch (...) {
        }
    }
}
}  // namespace ov
