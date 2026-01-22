// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

class ITagBuffer {
public:
    virtual std::string_view get_tag() const = 0;
    virtual size_t get_id() const = 0;
    virtual size_t get_offset() const = 0;
    virtual bool is_mapped() const = 0;
    virtual ~ITagBuffer() = default;
};

template <typename T>
class SharedBufferBase : public ov::AlignedBuffer, public ITagBuffer {
    private:
    template <typename U, typename = void>
    struct has_get_ptr : std::false_type {};

    template <typename U>
    struct has_get_ptr<U, std::void_t<decltype(std::declval<U>()->template get_ptr<char>())>> : std::true_type {};

    template<class U >
    static constexpr bool has_get_ptr_v = has_get_ptr<U>::value;

    using mmaped_memory_ptr = std::shared_ptr<ov::MappedMemory>;

    virtual const ITagBuffer* as_itag_buffer() const = 0;
public:
    SharedBufferBase(char* data, size_t size, const T& shared_object, const std::string& tag)
        : _shared_object(shared_object),
          m_tag(tag.empty() ? nullptr : std::make_shared<std::string>(tag)) {
        m_allocated_buffer = nullptr;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    SharedBufferBase(char* data, size_t size, const T& shared_object) : _shared_object(shared_object), m_tag(nullptr) {
        m_allocated_buffer = nullptr;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    virtual ~SharedBufferBase() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

    std::string_view get_tag() const override {
        return m_tag ? std::string_view(*m_tag) : std::string_view{};
    }

    std::size_t get_id() const override {
        return m_tag ? std::hash<std::string>{}(*m_tag) : 0;
    }

    bool is_mapped() const override {
        if constexpr (std::is_same<T, mmaped_memory_ptr>::value) {
            return true;
        } else if (auto itabuf = as_itag_buffer(); itabuf) {
            return itabuf->is_mapped();
        }
        return false;
    }

    size_t get_offset() const override {
        if constexpr (has_get_ptr_v<T>) {
            return m_aligned_buffer && _shared_object && _shared_object->template get_ptr<char>()
                       ? std::distance(_shared_object->template get_ptr<char>(), m_aligned_buffer)
                       : 0;
        } else {
            return 0;
        }
    }

protected:
    T _shared_object;
    std::shared_ptr<std::string> m_tag;
};


/// \brief SharedBuffer class to store pointer to pre-allocated buffer. Own the shared object.
template <typename T>
class SharedBuffer : public SharedBufferBase<T> {
protected:
    template <typename U>
    const ITagBuffer* as_itag_buffer_impl(const U& obj) const {
        using BareT = std::remove_cv_t<std::remove_reference_t<U>>;

        if constexpr (std::is_base_of_v<ITagBuffer, BareT>) {
            return &obj;
        }
        else if constexpr (std::is_pointer_v<BareT> && std::is_polymorphic_v<std::remove_pointer_t<BareT>>) {
            return dynamic_cast<ITagBuffer*>(obj);
        }
        else {
            return nullptr;
        }
    }

    const ITagBuffer* as_itag_buffer() const override {
        return as_itag_buffer_impl<T>(this->_shared_object);
    }

public:
    SharedBuffer(char* data, size_t size, const T& shared_object, const std::string& tag)
        : SharedBufferBase<T>(data, size, shared_object, tag) {}
    SharedBuffer(char* data, size_t size, const T& shared_object)
        : SharedBufferBase<T>(data, size, shared_object) {}
};


template <typename T>
class SharedBuffer<std::shared_ptr<T>> : public SharedBufferBase<std::shared_ptr<T>> {
protected:
    template <typename U>
    const ITagBuffer* as_itag_buffer_impl(const U& obj) const {
        if (!obj) {
            return nullptr;
        }
        if constexpr (std::is_polymorphic_v<typename U::element_type>) {
            return dynamic_cast<ITagBuffer*>(obj.get());
        } else {
            return nullptr;
        }
    }

    const ITagBuffer* as_itag_buffer() const override {
        if constexpr (std::is_same_v<T, void>) {
            return nullptr;
        } else {
            return as_itag_buffer_impl<std::shared_ptr<T>>(this->_shared_object);
        }
    }

public:
    SharedBuffer(char* data, size_t size, const std::shared_ptr<T>& shared_object, const std::string& tag)
        : SharedBufferBase<std::shared_ptr<T>>(data, size, shared_object, tag) {}
    SharedBuffer(char* data, size_t size, const std::shared_ptr<T>& shared_object)
        : SharedBufferBase<std::shared_ptr<T>>(data, size, shared_object) {}
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
