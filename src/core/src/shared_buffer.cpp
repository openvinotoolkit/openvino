// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_buffer.hpp"

namespace ov {
    SharedStreamBuffer::SharedStreamBuffer(char* data, size_t size) : m_data(data), m_size(size), m_offset(0) {}

    char* SharedStreamBuffer::data() {
        return m_data;
    }

    size_t SharedStreamBuffer::size() {
        return m_size;
    }

    std::streamsize SharedStreamBuffer::xsgetn(char* s, std::streamsize count) {
        auto real_count = std::min<std::streamsize>(m_size - m_offset, count);
        std::memcpy(s, m_data + m_offset, real_count);
        m_offset += real_count;
        return real_count;
    }

    std::streambuf::int_type SharedStreamBuffer::underflow() {
        return (m_size == m_offset) ? traits_type::eof() : traits_type::to_int_type(*(m_data + m_offset));
    }

    std::streambuf::int_type SharedStreamBuffer::uflow() {
        return (m_size == m_offset) ? traits_type::eof() : traits_type::to_int_type(*(m_data + m_offset++));
    }

    std::streamsize SharedStreamBuffer::showmanyc() {
        return m_size - m_offset;
    }

    OwningSharedStreamBuffer::OwningSharedStreamBuffer(char* data, size_t size, const std::shared_ptr<void>& shared_obj)
        : SharedStreamBuffer(data, size),
          m_shared_obj(shared_obj) {}
}  // namespace ov
