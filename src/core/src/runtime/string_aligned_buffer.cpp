// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/string_aligned_buffer.hpp"

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
StringAlignedBuffer::StringAlignedBuffer() : AlignedBuffer(), m_num_elements(0) {}

StringAlignedBuffer::StringAlignedBuffer(size_t num_elements, size_t byte_size, size_t alignment, bool initialize)
    : AlignedBuffer(byte_size, alignment),
      m_num_elements(num_elements) {
    OPENVINO_ASSERT(sizeof(std::string) * num_elements <= byte_size + alignment,
                    "Allocated memory of size " + std::to_string(byte_size) + " bytes is not enough to store " +
                        std::to_string(num_elements) + " std::string objects");
    if (initialize) {
        auto strings = reinterpret_cast<std::string*>(m_aligned_buffer);
        std::uninitialized_fill_n(strings, m_num_elements, std::string());
    }
}

StringAlignedBuffer::StringAlignedBuffer(StringAlignedBuffer&& other)
    : AlignedBuffer(std::move(other)),
      m_num_elements(other.m_num_elements) {
    other.m_num_elements = 0;
}

StringAlignedBuffer& StringAlignedBuffer::operator=(StringAlignedBuffer&& other) {
    if (this != &other) {
        if (m_allocated_buffer != nullptr) {
            delete[] m_allocated_buffer;
        }
        m_allocated_buffer = other.m_allocated_buffer;
        m_aligned_buffer = other.m_aligned_buffer;
        m_byte_size = other.m_byte_size;
        m_num_elements = other.m_num_elements;
        other.m_allocated_buffer = nullptr;
        other.m_aligned_buffer = nullptr;
        other.m_byte_size = 0;
        other.m_num_elements = 0;
    }
    return *this;
}

StringAlignedBuffer::~StringAlignedBuffer() {
    if (m_aligned_buffer) {
        auto strings = reinterpret_cast<std::string*>(m_aligned_buffer);
        for (size_t ind = 0; ind < m_num_elements; ++ind) {
            using std::string;
            strings[ind].~string();
        }
    }
}

}  // namespace ov
