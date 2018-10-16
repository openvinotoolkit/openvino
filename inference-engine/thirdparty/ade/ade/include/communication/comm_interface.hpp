// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef COMM_INTERFACE_HPP
#define COMM_INTERFACE_HPP

#include <memory>

#include "memory/memory_types.hpp"

#include "memory/memory_descriptor_ref.hpp"

namespace ade
{

class IDataBuffer;
class DataBufferView;

/// Data communication interface between different implementataions
class ICommChannel
{
public:
    virtual ~ICommChannel() = default;

    using Size = memory::DynMdSize;
    using View = memory::DynMdView<void>;
    using Span = memory::DynMdSpan;

    struct BufferDesc
    {
        /// Writers count for this buffer
        /// always greater than 0
        int  writersCount = 0;

        /// Readers count for this buffer
        /// always greater than 0
        int readersCount = 0;

        /// Reference to memory descriptor
        /// Will never be null
        MemoryDescriptorRef memoryRef;
    };

    struct BufferPrefs
    {
        /// Preferred alignment for buffer data
        /// dimensions count must be equal to buffer dimensions count
        /// Each element must be power of two
        /// Implementation can set elements to 1 if it doesn't care about alignment
        Size preferredAlignment;
    };

    /// Implementation must return buffer memory preferences
    ///
    /// @param desc Buffer description
    ///
    /// @returns Buffer memory preferences
    virtual BufferPrefs getBufferPrefs(const BufferDesc& desc) = 0;

    /// Implementation can return buffer object if it supports efficient implementation or null
    /// If all participants returned nulls buffer will be allocated by framework
    /// Caller takes ownership of this buffer
    ///
    /// @param desc Buffer description
    /// @param prefs Buffer memory preferences
    ///
    /// @returns Buffer object instance or null,  caller will take ownership of this object
    virtual std::unique_ptr<IDataBuffer> getBuffer(const BufferDesc& desc, const BufferPrefs& prefs) = 0;

    /// Framework will call the method for all participant after appropriate buffer was allocated
    /// (either by one of participants or by framework itself)
    /// Participant must store buffer object to be able to access its data during execution
    ///
    /// @param buffer Buffer object wrapper
    /// @param desc Buffer description
    virtual void setBuffer(const DataBufferView& buffer, const BufferDesc& desc) = 0;
};
}

#endif // COMM_INTERFACE_HPP
