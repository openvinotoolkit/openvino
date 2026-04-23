// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/file_view_buffer.hpp"

#include <fstream>

#include "openvino/core/memory_util.hpp"

namespace ov {
FileViewBuffer::FileViewBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size, size_t alignment)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_lazy_offset{offset},
      m_lazy_byte_size{byte_size},
      m_lazy_alignment{alignment},
      m_lazy_buffer{} {
    m_byte_size = byte_size;
}

void FileViewBuffer::load() const {
    if (m_lazy_buffer.empty()) {
        const size_t aligned_size = ((m_lazy_byte_size + m_lazy_alignment - 1) / m_lazy_alignment) * m_lazy_alignment;
        m_lazy_buffer.resize(aligned_size);
        const auto aligned_buffer =
            m_lazy_buffer.data() +
            util::align_padding_size(m_lazy_alignment, reinterpret_cast<size_t>(m_lazy_buffer.data()));

        try {
            std::ifstream file(m_file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open file: " + m_file_path.string());
            }
            file.seekg(m_lazy_offset);
            file.read(aligned_buffer, m_lazy_byte_size);
            if (!file) {
                throw std::runtime_error("Failed to read data from file: " + m_file_path.string());
            }
        } catch (...) {
            m_lazy_buffer.clear();
            throw;
        }

        // set derived members since AlignedBuffer isn't pure abstract
        m_aligned_buffer = aligned_buffer;
        m_byte_size = m_lazy_byte_size;
    }
}

void FileViewBuffer::release() const {
    m_lazy_buffer.clear();
    m_aligned_buffer = nullptr;
    m_byte_size = 0;
}
}  // namespace ov
