// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MEMORY_DESCRIPTOR_HPP
#define MEMORY_DESCRIPTOR_HPP

#include <util/memory_range.hpp>

#include "memory/memory_types.hpp"
#include "memory/memory_accessor.hpp"

namespace ade
{

class MemoryDescriptorRef;
class MemoryDescriptorView;

/// This class represents continuous buffer used in graph
/// It can have optional memory view, accessible to external users
class MemoryDescriptor final
{
public:
    /// MemoryDescriptor constructor
    ///
    /// @param element_size Data element size in bytes
    /// @param dims Memory dimensions in elements
    explicit MemoryDescriptor(size_t element_size,
                              const memory::DynMdSize& dims);
    MemoryDescriptor(const MemoryDescriptor&) = delete;
    MemoryDescriptor& operator=(const MemoryDescriptor&) = delete;
    ~MemoryDescriptor();

    /// Add events listener
    ///
    /// @param listener Listener to be added, must not be null
    /// Same listener must not be added twice
    void addListener(IMemoryAccessListener* listener);

    /// Remove events listener
    ///
    /// @param listener Listener to be removed, must not be null
    /// Listener must be previously added via addListener call
    void removeListener(IMemoryAccessListener* listener);

    using AccessHandle = MemoryAccessor::AccessHandle;

    /// Notify all listeners about memory access
    ///
    /// @param span Span that represents a ROI of current view to be accessed
    /// @param accessType Access type for this view, it must be Read, Write or ReadWrite
    /// @returns Handle which later will be passed to commit
    AccessHandle access(const memory::DynMdSpan& span, MemoryAccessType accessType);

    /// Notify all listeners about memory commit
    ///
    /// @param handle Handle returned from successuful access call
    void commit(AccessHandle handle);

    /// Get memory dimensions
    ///
    /// @returns Dimensions
    const memory::DynMdSize& dimensions() const;

    /// Get buffer element size
    ///
    /// @returns Size in bytes, always greater than 0
    std::size_t elementSize() const;

    /// Update externally accessible memory view and notify all listeners
    ///
    /// @param view New memory view
    void setExternalView(const memory::DynMdView<void>& view);

    /// Returns externally accessible memory view if any
    ///
    /// @returns View into externally accessible memory or null view
    memory::DynMdView<void> getExternalView() const;

private:
    friend class MemoryDescriptorView;

    const size_t m_elementSize;
    const memory::DynMdSize m_dims;
    memory::DynMdView<void> m_externalView;

    MemoryAccessor m_accessor;
};

}

#endif // MEMORY_DESCRIPTOR_HPP
