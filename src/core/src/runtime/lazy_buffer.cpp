// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <fstream>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size, size_t alignment)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_alignment{alignment},
      m_lazy_buffer{} {
    OPENVINO_ASSERT(util::file_exists(m_file_path), "File does not exist: ", m_file_path.string());
    const auto file_size = util::file_size(m_file_path);
    OPENVINO_ASSERT(file_size >= 0 && static_cast<size_t>(file_size) >= m_offset + byte_size,
                    "File size is smaller than the requested view (file size: ",
                    file_size,
                    ", requested offset: ",
                    m_offset,
                    ", requested byte size: ",
                    byte_size,
                    ").");
    m_byte_size = byte_size;
}

void LazyBuffer::load() const {
    if (m_lazy_buffer.empty() && m_byte_size > 0) {
        const size_t aligned_size = ((m_byte_size + m_alignment - 1) / m_alignment) * m_alignment;
        m_lazy_buffer.resize(aligned_size);

        const auto allocated_buffer = m_lazy_buffer.data();
        m_aligned_buffer =
            allocated_buffer + util::align_padding_size(m_alignment, reinterpret_cast<size_t>(allocated_buffer));

        try {
            std::ifstream file(m_file_path, std::ios::binary);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path.string());
            file.seekg(m_offset).read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path.string());
        } catch (...) {
            m_aligned_buffer = {};
            m_lazy_buffer.clear();
            throw;
        }
    }
}

void LazyBuffer::unload() {
    m_aligned_buffer = {};
    m_lazy_buffer.clear();
}
}  // namespace ov
