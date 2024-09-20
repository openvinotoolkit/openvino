// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {

/// \brief SharedBuffer class to store pointer to pre-acclocated buffer. Own the shared object.
template <typename T>
class SharedBuffer : public ov::AlignedBuffer {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_allocated_buffer = data;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    virtual ~SharedBuffer() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

private:
    T _shared_object;
};

/// \brief SharedStreamBuffer class to store pointer to pre-acclocated buffer and provide streambuf interface.
class SharedStreamBuffer : public std::streambuf {
public:
    SharedStreamBuffer(char* data, size_t size) : m_data(data), m_size(size), m_offset(0) {}

protected:
    std::streamsize xsgetn(char* s, std::streamsize count) override {
        auto real_count = std::min<std::streamsize>(m_size - m_offset, count);
        std::memcpy(s, m_data + m_offset, real_count);
        m_offset += real_count;
        return real_count;
    }

    int_type underflow() override {
        return (m_size == m_offset) ? traits_type::eof() : traits_type::to_int_type(*(m_data + m_offset));
    }

    int_type uflow() override {
        return (m_size == m_offset) ? traits_type::eof() : traits_type::to_int_type(*(m_data + m_offset++));
    }

    std::streamsize showmanyc() override {
        return m_size - m_offset;
    }

    size_t m_size;
    char* m_data;
    size_t m_offset;
};

/// \brief OwningSharedStreamBuffer is a SharedStreamBuffer which owns its shared object. Can return AlignedBuffer to
/// shared memory
class OwningSharedStreamBuffer : public SharedStreamBuffer {
public:
    template <typename T>
    OwningSharedStreamBuffer(char* data, size_t size, const T& shared_object)
        : SharedStreamBuffer(data, size),
          m_alligned_buffer(std::make_shared<SharedBuffer<T>>(data, size, shared_object)) {}

    std::shared_ptr<AlignedBuffer> get_aligned_buffer() {
        return m_alligned_buffer;
    }

protected:
    std::shared_ptr<AlignedBuffer> m_alligned_buffer;
};

}  // namespace ov
