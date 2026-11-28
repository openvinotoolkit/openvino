// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/util/memory.hpp"

namespace ov {

class AlignedBuffer;

class OPENVINO_API IBufferDescriptor {
public:
    virtual size_t get_id() const = 0;
    virtual size_t get_offset() const = 0;
    virtual std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const = 0;
    virtual ~IBufferDescriptor();
};

/// \brief Allocates a block of memory which starts at an aligned address determined by \p alignment.
class OPENVINO_API AlignedBuffer : public util::IMutableBuffer {
public:
    // Allocator objects and the allocation interfaces are owned by the
    // creators of AlignedBuffers. They need to ensure that the lifetime of
    // allocator exceeds the lifetime of this AlignedBuffer.
    AlignedBuffer(size_t byte_size, size_t alignment = 64);

    AlignedBuffer();
    ~AlignedBuffer() override;

    AlignedBuffer(AlignedBuffer&& other);
    AlignedBuffer& operator=(AlignedBuffer&& other);

    const std::byte* data() const noexcept override final {
        return m_aligned_buffer;
    }
    std::byte* data() noexcept override final {
        return m_aligned_buffer;
    }
    size_t size() const noexcept override final {
        return m_byte_size;
    }
    const util::MemoryProperties& get_properties() const noexcept override;

    void* get_ptr(size_t offset) const {
        return m_aligned_buffer + offset;
    }
    void* get_ptr() {
        return m_aligned_buffer;
    }
    const void* get_ptr() const {
        return m_aligned_buffer;
    }
    template <typename T>
    T* get_ptr() {
        return reinterpret_cast<T*>(m_aligned_buffer);
    }
    template <typename T>
    const T* get_ptr() const {
        return reinterpret_cast<const T*>(m_aligned_buffer);
    }

    template <typename T>
    explicit operator T*() {
        return get_ptr<T>();
    }

    virtual std::shared_ptr<IBufferDescriptor> get_descriptor() const;

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void hint_evict() noexcept override;

    util::PrefetchToken hint_prefetch() noexcept override;

protected:
    virtual void hint_evict(size_t offset, size_t size) noexcept;
    static void invoke_evict(AlignedBuffer& buffer, size_t offset, size_t size) noexcept;

    virtual util::PrefetchToken hint_prefetch_async(size_t offset, size_t size) const noexcept;
    static util::PrefetchToken invoke_hint_prefetch_async(const AlignedBuffer& buffer,
                                                          size_t offset,
                                                          size_t size) noexcept;

    std::byte* m_aligned_buffer;
    size_t m_byte_size;
};

template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>
    : public DirectValueAccessor<std::shared_ptr<ov::AlignedBuffer>> {
public:
    AttributeAdapter(std::shared_ptr<ov::AlignedBuffer>& value);
    ~AttributeAdapter() override;
    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>");
};
}  // namespace ov
