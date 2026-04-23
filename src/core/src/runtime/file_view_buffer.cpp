// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/file_view_buffer.hpp"

#include <fstream>

namespace ov {
FileViewBuffer::FileViewBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_lazy_offset{offset},
      m_lazy_byte_size{byte_size},
      m_lazy_buffer{} {
    m_byte_size = byte_size;
}

void FileViewBuffer::load() const {
    if (m_lazy_buffer.empty()) {
        std::ifstream file(m_file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + m_file_path.string());
        }
        file.seekg(m_lazy_offset);
        m_lazy_buffer.resize(m_lazy_byte_size);
        file.read(m_lazy_buffer.data(), m_lazy_byte_size);
        if (!file) {
            throw std::runtime_error("Failed to read data from file: " + m_file_path.string());
        }
        m_aligned_buffer = m_lazy_buffer.data();
        m_byte_size = m_lazy_byte_size;
    }
}

void FileViewBuffer::release() const {
    m_lazy_buffer.clear();
    m_aligned_buffer = nullptr;
    m_byte_size = 0;
}
}  // namespace ov
