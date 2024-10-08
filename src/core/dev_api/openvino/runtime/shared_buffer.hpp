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
///  Can return ptr to shared memory and its size
class SharedStreamBuffer : public std::streambuf {
public:
    SharedStreamBuffer(char* data, size_t size);

    // get data ptr and its size
    char* data();
    size_t size();

protected:
    // override std::streambuf methods
    std::streamsize xsgetn(char* s, std::streamsize count) override;
    int_type underflow() override;
    int_type uflow() override;
    std::streamsize showmanyc() override;

    char* m_data;
    size_t m_size;
    size_t m_offset;
};

/// \brief OwningSharedStreamBuffer is a SharedStreamBuffer which owns its shared object. 
class OwningSharedStreamBuffer : public SharedStreamBuffer {
public:
    OwningSharedStreamBuffer(char* data, size_t size, const std::shared_ptr<void>& shared_obj);

protected:
    std::shared_ptr<void> m_shared_obj;
};

}  // namespace ov
