// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef ADE_MEMORYACCESSLISTENER_HPP
#define ADE_MEMORYACCESSLISTENER_HPP

#include <memory>

#include "util/assert.hpp"

#include "memory/memory_types.hpp"

namespace ade
{

class MemoryDescriptor;

enum class MemoryAccessType
{
    NoAccess,
    Read,
    Write,
    ReadWrite,
};

/// Listener used to notify internal objects
/// when graph managed data is being accessed extenally
class IMemoryAccessListener
{
public:
    virtual ~IMemoryAccessListener() {}

    using AccessHandle = void*;

    struct AccessHandleDeleter
    {
        IMemoryAccessListener* listener;
        void operator()(AccessHandle handle)
        {
            ASSERT(nullptr != listener);
            listener->commitImpl(handle);
        }
    };

    using AccessHandlePtr = std::unique_ptr<void, AccessHandleDeleter>;

    /// This method is called just before external user will be able to access data
    /// so listener can do required preparations (e.g. do OpenCL map or copy data from device)
    /// Listener can report errors via exceptions
    ///
    /// @param desc The memory descriptor to be accessed
    /// @param span Span that represents a ROI of current view to be accessed
    /// @param accessType Access type for this view, it can be Read, Write or ReadWrite
    /// @returns Listener specific handle which later will be passed to commitImpl
    /// Framework doesn't check this return value in any way
    virtual AccessHandle accessImpl(const MemoryDescriptor& desc, const memory::DynMdSpan& span, MemoryAccessType accessType) = 0;

    /// This method is called immediately after external user finished access to data
    /// This method must never throw
    ///
    /// @param handle Handle returned from successuful accessImpl call
    virtual void commitImpl(AccessHandle handle) = 0;

    /// This method called when externally accessible data view is about to change
    /// Listener must drop all references to old data and use new
    /// Listener can report errors via exceptions
    ///
    /// @param old_view Previous data, can be null if data wasn't externally accessible previously
    /// @param new_view New data, can be null
    virtual void memoryViewChangedImpl(const memory::DynMdView<void>& old_view,
                                       const memory::DynMdView<void>& new_view) = 0;

    /// This method is called when external memory descriptor is destroyed,
    /// so the listener can remove any objects created for it (OpenCL buffers, etc.)
    virtual void memoryDescriptorDestroyedImpl() = 0;

    AccessHandlePtr access(const MemoryDescriptor& desc, const memory::DynMdSpan& span, MemoryAccessType accessType)
    {
        return AccessHandlePtr(accessImpl(desc, span, accessType),
                               AccessHandleDeleter{this});
    }
};

}

#endif // ADE_MEMORYACCESSLISTENER_HPP
