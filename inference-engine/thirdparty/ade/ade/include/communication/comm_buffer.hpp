// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef COMM_BUFFER_HPP
#define COMM_BUFFER_HPP

#include "util/assert.hpp"

#include "memory/memory_types.hpp"

namespace ade
{

/// Interface to access data objects provided by implementations
/// Beware: Methods of this interface can be called from unspecified threads
/// Implementations must provide thread-safety
class IDataBuffer
{
public:
    virtual ~IDataBuffer() = default;

    using Size = memory::DynMdSize;
    using View = memory::DynMdView<void>;
    using Span = memory::DynMdSpan;

    enum Access
    {
        Read,
        Write
    };

    struct MapId
    {
        /// View into buffer memory
        View view;

        /// Implementation-specific handle
        std::uintptr_t handle;
    };

    /// Access buffer memory
    ///
    /// @param span Area which user want to access
    /// @param access Access type (read/write)
    ///
    /// @returns MapId object containg host accessible memory view and
    /// some implementation-specific opaque handle
    virtual MapId map(const Span& span, Access access) = 0;

    /// Finish accessing buffer memory
    ///
    /// @param id MapId object obtained from successful call to IDataBuffer::map
    virtual void unmap(const MapId& id) = 0;

    /// This method must be called when implementation finished writing to the buffer
    /// during current execution cycle
    ///
    /// It is invalid to access memory for writing for the specified span
    /// during current execution cycle after call to this method
    ///
    /// It is invalid to access memory for reading for the specified span
    /// during current execution cycle before call to this method
    ///
    /// Each producer must call this method for the span received in ICommChannel::setBuffer
    /// before consumers can access it contents
    ///
    /// @param span Span received in ICommChannel::setBuffer
    virtual void finalizeWrite(const Span& span) = 0;

    /// This method must be called when implementation finished reading from the buffer
    /// during current execution cycle
    ///
    /// It is invalid to access memory for reading or writing for the specified span
    /// during current execution cycle after call to this method
    ///
    /// Each consumer must call this method for the span received in ICommChannel::setBuffer
    /// before finishing execution
    ///
    /// @param span Span received in ICommChannel::setBuffer
    virtual void finalizeRead(const Span& span) = 0;

    /// Returns memory alignment, guaranteed by current implementation
    /// Can be called during compilation and during execution
    ///
    /// @param span Area which alignment we want to query
    ///
    /// @returns Alignment guaranteed for each dimension of specified area
    virtual Size alignment(const Span& span) = 0;
};

/// Buffer wrapper which automatically applies offset to each buffer method call
class DataBufferView final
{
    using Span = memory::DynMdSpan;
    using Size = memory::DynMdSize;

    IDataBuffer* m_buffer = nullptr;
    Span m_span;

    Span fixSpan(const Span& span) const
    {
        Span ret = span + m_span.origin();
        for (auto i: util::iota(m_span.dims_count()))
        {
            ASSERT(ret[i].begin >= m_span[i].begin);
            ASSERT(ret[i].end   <= m_span[i].end);
        }
        return ret;
    }
public:
    DataBufferView() = default;
    DataBufferView(std::nullptr_t) {}
    DataBufferView(IDataBuffer* buff, const Span& span):
        m_buffer(buff), m_span(span) {}

    DataBufferView(const DataBufferView&) = default;
    DataBufferView& operator=(const DataBufferView&) = default;
    DataBufferView& operator=(std::nullptr_t)
    {
        m_buffer = nullptr;
        m_span = Span{};
        return *this;
    }

    /// Get original buffer
    ///
    /// @returns Buffer from which this view was created or null for default constructed view
    IDataBuffer* getBuffer() const
    {
        return m_buffer;
    }

    /// Get span of this buffer wrapper
    ///
    /// @returns Which part of the original buffer this view references to
    Span getSpan() const
    {
        return m_span;
    }

    /// Applies offset to span and calls IDataBuffer::map
    IDataBuffer::MapId map(const Span& span, IDataBuffer::Access access)
    {
        ASSERT(nullptr != m_buffer);
        return m_buffer->map(fixSpan(span), access);
    }

    /// Calls IDataBuffer::unmap
    void unmap(const IDataBuffer::MapId& id)
    {
        ASSERT(nullptr != m_buffer);
        m_buffer->unmap(id);
    }

    /// Calls IDataBuffer::finalizeWrite with current span
    void finalizeWrite()
    {
        ASSERT(nullptr != m_buffer);
        m_buffer->finalizeWrite(m_span);
    }

    /// Calls IDataBuffer::finalizeRead with current span
    void finalizeRead()
    {
        ASSERT(nullptr != m_buffer);
        m_buffer->finalizeRead(m_span);
    }

    /// Applies offset to span and calls IDataBuffer::alignment
    Size alignment(const Span& span)
    {
        ASSERT(nullptr != m_buffer);
        return m_buffer->alignment(fixSpan(span));
    }
};

class DataBufferMapper final
{
    DataBufferView& m_view;
    IDataBuffer::MapId m_map_id;
public:
    DataBufferMapper(DataBufferView& view, const memory::DynMdSpan& span, IDataBuffer::Access access):
        m_view(view), m_map_id(view.map(span, access)) {}

    DataBufferMapper(const DataBufferMapper&) = delete;
    DataBufferMapper& operator=(const DataBufferMapper&) = delete;

    ~DataBufferMapper()
    {
        m_view.unmap(m_map_id);
    }

    memory::DynMdView<void> view() const
    {
        ASSERT(nullptr != m_map_id.view);
        return m_map_id.view;
    }
};

}

#endif // COMM_BUFFER_HPP
