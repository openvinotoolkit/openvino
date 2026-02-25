// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {

/// \brief SharedBuffer class to store pointer to pre-allocated buffer. Own the shared object.
template <typename T>
class SharedBuffer : public ov::AlignedBuffer {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_allocated_buffer = nullptr;
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

/// \brief SharedStreamBuffer class to store pointer to pre-allocated buffer and provide streambuf interface.
///  Can return ptr to shared memory and its size
class SharedStreamBuffer : public std::streambuf {
public:
    SharedStreamBuffer(const char* data, size_t size) : m_data(data), m_size(size), m_offset(0) {}
    explicit SharedStreamBuffer(const void* data, size_t size)
        : SharedStreamBuffer(reinterpret_cast<const char*>(data), size) {}

protected:
    // override std::streambuf methods
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

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(pos, std::ios_base::beg, which);
    }

    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override {
        if (which != std::ios_base::in) {
            return pos_type(off_type(-1));
        }

        size_t new_offset;
        switch (dir) {
        case std::ios_base::beg:
            new_offset = off;
            break;
        case std::ios_base::cur:
            new_offset = m_offset + off;
            break;
        case std::ios_base::end:
            new_offset = m_size + off;
            break;
        default:
            return pos_type(off_type(-1));
        }

        // Check bounds
        if (new_offset > m_size) {
            return pos_type(off_type(-1));
        }

        m_offset = new_offset;
        return pos_type(m_offset);
    }

    const char* m_data;
    const size_t m_size;
    size_t m_offset;
};

/// \brief OwningSharedStreamBuffer is a SharedStreamBuffer which owns its shared object.
class OwningSharedStreamBuffer : public SharedStreamBuffer {
public:
    OwningSharedStreamBuffer(std::shared_ptr<ov::AlignedBuffer> buffer)
        : SharedStreamBuffer(static_cast<char*>(buffer->get_ptr()), buffer->size()),
          m_shared_obj(buffer) {}

    std::shared_ptr<ov::AlignedBuffer> get_buffer() {
        return m_shared_obj;
    }

protected:
    std::shared_ptr<ov::AlignedBuffer> m_shared_obj;
};
}  // namespace ov
