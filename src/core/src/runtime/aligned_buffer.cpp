// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/aligned_buffer.hpp"

#include <algorithm>
#include <memory>

namespace ov {
AlignedBuffer::AlignedBuffer() : m_allocated_buffer(nullptr), m_aligned_buffer(nullptr), m_byte_size(0) {}

AlignedBuffer::AlignedBuffer(size_t byte_size, size_t alignment) : m_byte_size(byte_size) {
    m_byte_size = std::max<size_t>(1, byte_size);
    size_t allocation_size = m_byte_size + alignment;
    m_allocated_buffer = new char[allocation_size];
    m_aligned_buffer = m_allocated_buffer;
    size_t mod = (alignment != 0) ? reinterpret_cast<size_t>(m_aligned_buffer) % alignment : 0;

    if (mod != 0) {
        m_aligned_buffer += (alignment - mod);
    }
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other)
    : m_allocated_buffer(other.m_allocated_buffer),
      m_aligned_buffer(other.m_aligned_buffer),
      m_byte_size(other.m_byte_size) {
    other.m_allocated_buffer = nullptr;
    other.m_aligned_buffer = nullptr;
    other.m_byte_size = 0;
}

AlignedBuffer::~AlignedBuffer() {
    if (m_allocated_buffer != nullptr) {
        delete[] m_allocated_buffer;
    }
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) {
    if (this != &other) {
        if (m_allocated_buffer != nullptr) {
            delete[] m_allocated_buffer;
        }
        m_allocated_buffer = other.m_allocated_buffer;
        m_aligned_buffer = other.m_aligned_buffer;
        m_byte_size = other.m_byte_size;
        other.m_allocated_buffer = nullptr;
        other.m_aligned_buffer = nullptr;
        other.m_byte_size = 0;
    }
    return *this;
}

AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>::AttributeAdapter(std::shared_ptr<ov::AlignedBuffer>& value)
    : DirectValueAccessor<std::shared_ptr<ov::AlignedBuffer>>(value) {}

AttributeAdapter<std::shared_ptr<AlignedBuffer>>::~AttributeAdapter() = default;
}  // namespace ov
