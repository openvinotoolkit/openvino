// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
OPENVINO_API std::shared_ptr<IBufferDescriptor>
create_base_descriptor(size_t id, size_t offset, const std::shared_ptr<ov::AlignedBuffer>& source_buffer);

namespace detail {
OPENVINO_API std::shared_ptr<IBufferDescriptor> create_mmap_descriptor(const std::shared_ptr<ov::MappedMemory>& mmap);
}  // namespace detail

template <typename T>
class SharedBufferBase : public ov::AlignedBuffer {
public:
    std::shared_ptr<IBufferDescriptor> get_descriptor() const override {
        return m_descriptor;
    }

    const util::MemoryProperties& get_properties() const noexcept override {
        return m_properties;
    }

    std::optional<uint64_t> get_id() const noexcept override {
        return m_id;
    }

    virtual ~SharedBufferBase() {
        m_aligned_buffer = nullptr;
        m_byte_size = 0;
    }

    virtual void hint_evict() noexcept override {
        hint_evict(m_properties.offset, m_byte_size);
    }

    util::PrefetchToken hint_prefetch() noexcept override {
        return hint_prefetch_async(m_properties.offset, m_byte_size);
    }

protected:
    // Shared object implementing the common util::IBuffer interface (MappedMemory, AlignedBuffer, ...).
    template <typename U>
    struct is_buffer_ptr : std::false_type {};
    template <typename U>
    struct is_buffer_ptr<std::shared_ptr<U>> : std::is_base_of<util::IBuffer, U> {};
    template <typename U>
    static constexpr bool is_buffer_ptr_v = is_buffer_ptr<U>::value;

    // Source ID and root-relative offset of [data, data + size) when not given by a descriptor.
    static std::optional<uint64_t> source_id(const T& object) noexcept {
        if constexpr (is_buffer_ptr_v<T>) {
            if (object) {
                return object->get_id();
            }
        }
        return std::nullopt;
    }

    static size_t source_offset(const T& object, const std::byte* data) noexcept {
        if constexpr (is_buffer_ptr_v<T>) {
            if (object && data >= object->data() && data <= object->data() + object->size()) {
                return object->get_properties().offset + static_cast<size_t>(data - object->data());
            }
        }
        return 0;
    }

    // The offset is relative to the root source buffer, so it is forwarded to the parents unchanged.
    virtual void hint_evict(size_t offset, size_t size) noexcept override {
        if constexpr (is_buffer_ptr_v<T>) {
            if (m_shared_object) {
                auto* object = m_shared_object.get();
                if (auto* mapped = dynamic_cast<ov::MappedMemory*>(object)) {
                    mapped->hint_evict(offset, size);
                } else if (auto* aligned = dynamic_cast<ov::AlignedBuffer*>(object)) {
                    invoke_evict(*aligned, offset, size);
                } else if (offset == object->get_properties().offset && size == object->size()) {
                    object->hint_evict();  // only whole-buffer hints are available
                }
            }
        }
    }

    util::PrefetchToken hint_prefetch_async(size_t offset, size_t size) const noexcept override {
        if constexpr (is_buffer_ptr_v<T>) {
            if (m_shared_object) {
                auto* object = m_shared_object.get();
                if (auto* mapped = dynamic_cast<ov::MappedMemory*>(object)) {
                    return mapped->hint_prefetch_async(offset, size);
                } else if (auto* aligned = dynamic_cast<ov::AlignedBuffer*>(object)) {
                    return invoke_hint_prefetch_async(*aligned, offset, size);
                } else if (offset == object->get_properties().offset && size == object->size()) {
                    return object->hint_prefetch();  // only whole-buffer hints are available
                }
            }
        }
        return {};
    }

    // protected to not create SharedBufferBase directly
    SharedBufferBase(std::byte* data,
                     size_t size,
                     const T& shared_object,
                     const std::shared_ptr<IBufferDescriptor>& descriptor)
        : m_shared_object{shared_object},
          m_source_buffer{descriptor ? descriptor->get_source_buffer() : nullptr},
          m_descriptor{[&] {
              if (descriptor) {
                  uintptr_t offset{};
                  if (m_source_buffer) {
                      const auto src_start = reinterpret_cast<uintptr_t>(m_source_buffer->get_ptr());
                      const auto src_end = src_start + static_cast<uintptr_t>(m_source_buffer->size());
                      const auto current = reinterpret_cast<uintptr_t>(data);
                      const auto data_in_src_range = (current >= src_start) &&
                                                     !ov::util::add_overflow(current, size, offset) &&
                                                     (offset <= src_end);
                      OPENVINO_ASSERT(data_in_src_range, "SharedBuffer data range is outside source buffer range");
                      offset = current - src_start;
                  }
                  return create_base_descriptor(descriptor->get_id(), offset, m_source_buffer);
              } else {
                  return std::shared_ptr<IBufferDescriptor>{};
              }
          }()},
          m_properties{m_descriptor ? m_descriptor->get_offset() : source_offset(shared_object, data)},
          m_id{descriptor ? std::optional<uint64_t>{descriptor->get_id()} : std::nullopt} {
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    SharedBufferBase(std::byte* data, size_t size, const T& shared_object)
        : m_shared_object{shared_object},
          m_source_buffer{},
          m_descriptor{},
          m_properties{source_offset(shared_object, data)},
          m_id{source_id(shared_object)} {
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    // Owns the underlying data and keeps it alive for the lifetime of this buffer
    T m_shared_object;
    // Points to the root AlignedBuffer used for offset calculation and
    // accessible externally via get_descriptor()->get_source_buffer();
    // may or may not reference the same data as m_shared_object
    std::shared_ptr<ov::AlignedBuffer> m_source_buffer;
    std::shared_ptr<IBufferDescriptor> m_descriptor;
    util::MemoryProperties m_properties;
    std::optional<uint64_t> m_id;
};

template <typename T>
class SharedBuffer : public SharedBufferBase<T> {
    static std::shared_ptr<IBufferDescriptor> get_or_make_descriptor(const T& shared_object) {
        if constexpr (std::is_same_v<T, std::shared_ptr<ov::MappedMemory>>) {
            return detail::create_mmap_descriptor(shared_object);
        } else if constexpr (SharedBufferBase<T>::template is_buffer_ptr_v<T>) {
            if (auto mapped = std::dynamic_pointer_cast<ov::MappedMemory>(shared_object)) {
                return detail::create_mmap_descriptor(mapped);
            } else if (auto aligned = std::dynamic_pointer_cast<ov::AlignedBuffer>(shared_object)) {
                return aligned->get_descriptor();
            } else {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

public:
    SharedBuffer(std::byte* data,
                 size_t size,
                 const T& shared_object,
                 const std::shared_ptr<IBufferDescriptor>& descriptor)
        : SharedBufferBase<T>(data, size, shared_object, descriptor) {}

    SharedBuffer(std::byte* data, size_t size, const T& shared_object)
        : SharedBuffer(data, size, shared_object, get_or_make_descriptor(shared_object)) {
        this->m_id = SharedBufferBase<T>::source_id(shared_object);
    }

    SharedBuffer(char* data, size_t size, const T& shared_object, const std::shared_ptr<IBufferDescriptor>& descriptor)
        : SharedBuffer(reinterpret_cast<std::byte*>(data), size, shared_object, descriptor) {}

    SharedBuffer(char* data, size_t size, const T& shared_object)
        : SharedBuffer(reinterpret_cast<std::byte*>(data), size, shared_object) {}
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
}  // namespace ov
