// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>
#include <type_traits>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
class OPENVINO_API IBufferDescriptor {
public:
    virtual size_t get_id() const = 0;
    virtual size_t get_offset() const = 0;
    virtual std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const = 0;
    virtual ~IBufferDescriptor() = default;
};

template <typename T>
class SharedBufferBase : public ov::AlignedBuffer {
    class SharedBufferDescriptor : public IBufferDescriptor {
    public:
        SharedBufferDescriptor(size_t id, size_t offset, const std::shared_ptr<ov::AlignedBuffer>& source_buffer)
            : m_id(id),
              m_offset(offset),
              m_source_buffer(source_buffer) {}

        size_t get_id() const override {
            return m_id;
        }

        size_t get_offset() const override {
            return m_offset;
        }

        std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
            return m_source_buffer.lock();
        }

    private:
        size_t m_id = 0;
        size_t m_offset = 0;
        std::weak_ptr<ov::AlignedBuffer> m_source_buffer;
    };

public:
    SharedBufferBase(char* data,
                     size_t size,
                     const T& shared_object,
                     const std::shared_ptr<IBufferDescriptor>& descriptor)
        : _shared_object(shared_object),
          m_source_buffer(descriptor ? descriptor->get_source_buffer() : nullptr) {
        m_allocated_buffer = nullptr;
        m_aligned_buffer = data;
        m_byte_size = size;
        if (descriptor) {
            m_descriptor =
                std::make_shared<SharedBufferDescriptor>(descriptor->get_id(), get_offset(), m_source_buffer);
        }
    }

    SharedBufferBase(char* data, size_t size, const T& shared_object)
        : _shared_object(shared_object),
          m_source_buffer(nullptr) {
        m_allocated_buffer = nullptr;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    virtual ~SharedBufferBase() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

    std::shared_ptr<IBufferDescriptor> get_descriptor() const override {
        return m_descriptor;
    }

protected:
    T _shared_object;
    std::shared_ptr<ov::AlignedBuffer> m_source_buffer;
    mutable std::shared_ptr<IBufferDescriptor> m_descriptor;

    virtual size_t get_offset() const {
        if (m_source_buffer) {
            return std::distance(static_cast<char*>(m_source_buffer->get_ptr()), m_aligned_buffer);
        }
        return 0;
    }
};

template <typename T, typename = void>
class SharedBuffer : public SharedBufferBase<T> {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object, const std::shared_ptr<IBufferDescriptor>& descriptor)
        : SharedBufferBase<T>(data, size, shared_object, descriptor) {}

    SharedBuffer(char* data, size_t size, const T& shared_object) : SharedBufferBase<T>(data, size, shared_object) {}
};

template <>
class SharedBuffer<std::shared_ptr<ov::MappedMemory>> : public SharedBufferBase<std::shared_ptr<ov::MappedMemory>> {
    class MMapDescriptor : public IBufferDescriptor {
    public:
        MMapDescriptor(const std::weak_ptr<ov::MappedMemory>& mem, uint64_t id) : m_mem(mem), m_id(id) {}
        uint64_t get_id() const override {
            return m_id;
        }
        size_t get_offset() const override {
            return 0;
        }
        std::shared_ptr<AlignedBuffer> get_source_buffer() const override {
            if (auto mmap = m_mem.lock()) {
                // Use 3-arg constructor (no descriptor) to avoid infinite recursion:
                return std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(mmap->data(), mmap->size(), mmap);
            }
            return nullptr;
        }

    protected:
        std::weak_ptr<ov::MappedMemory> m_mem;
        uint64_t m_id;
    };

public:
    SharedBuffer(char* data, size_t size, const std::shared_ptr<ov::MappedMemory>& shared_object)
        : SharedBufferBase<std::shared_ptr<ov::MappedMemory>>(
              data,
              size,
              shared_object,
              std::make_shared<MMapDescriptor>(shared_object, shared_object->get_id())) {}
};

template <typename T>
class SharedBuffer<std::shared_ptr<T>, std::enable_if_t<std::is_base_of_v<ov::AlignedBuffer, T>>>
    : public SharedBufferBase<std::shared_ptr<T>> {
public:
    SharedBuffer(char* data,
                 size_t size,
                 const std::shared_ptr<T>& shared_object,
                 const std::shared_ptr<IBufferDescriptor>& descriptor)
        : SharedBufferBase<std::shared_ptr<T>>(data, size, shared_object, descriptor) {}

    SharedBuffer(char* data, size_t size, const std::shared_ptr<T>& shared_object)
        : SharedBufferBase<std::shared_ptr<T>>(data,
                                               size,
                                               shared_object,
                                               shared_object ? shared_object->get_descriptor() : nullptr) {}

    SharedBuffer(const std::shared_ptr<T>& shared_object, size_t offset, size_t size)
        : SharedBufferBase<std::shared_ptr<T>>(static_cast<char*>(shared_object->get_ptr()) + offset,
                                               size,
                                               shared_object,
                                               shared_object->get_descriptor()) {}
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
