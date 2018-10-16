// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MEMORY_DESCRIPTOR_REF_HPP
#define MEMORY_DESCRIPTOR_REF_HPP

#include <iosfwd>

#include "memory/memory_types.hpp"

namespace ade
{

class MemoryDescriptor;
class MemoryDescriptorView;

/// Lghtweight reference into MemoryDescriptorView
/// It can reference to a part of MemoryDescriptorView
class MemoryDescriptorRef final
{
public:
    /// Constructs empty MemoryDescriptorRef
    MemoryDescriptorRef();

    /// Consructs MemoryDescriptorRef covering entire view
    ///
    /// @param view Source view
    MemoryDescriptorRef(MemoryDescriptorView& view);

    /// Consructs MemoryDescriptorRef covering part view
    ///
    /// @param view Source view
    /// @param span Span into descriptor view, must be inside view dimensions
    MemoryDescriptorRef(MemoryDescriptorView& view, const memory::DynMdSpan& span);
    ~MemoryDescriptorRef();

    MemoryDescriptorRef(const MemoryDescriptorRef&) = default;
    MemoryDescriptorRef(MemoryDescriptorRef&&) = default;
    MemoryDescriptorRef& operator=(const MemoryDescriptorRef&) = default;
    MemoryDescriptorRef& operator=(MemoryDescriptorRef&&) = default;

    /// Get source view
    ///
    /// @returns Source view
    MemoryDescriptorView* getView();

    /// Get source view
    ///
    /// @returns Source view
    const MemoryDescriptorView* getView() const;

    /// Get source descriptor
    ///
    /// @returns Source descriptor
    MemoryDescriptor* getDescriptor();

    /// Get source descriptor
    ///
    /// @returns Source descriptor
    const MemoryDescriptor* getDescriptor() const;

    /// Get span into current view
    ///
    /// @returns span
    const memory::DynMdSpan& span() const;

    /// Get span size
    ///
    /// @returns size
    memory::DynMdSize size() const;

    /// Get buffer element size
    ///
    /// @returns Size in bytes
    std::size_t elementSize() const;

    /// Get span into parent ref
    ///
    /// @returns span
    memory::DynMdSpan originSpan() const;

    /// Returns externally accessible memory view if any
    ///
    /// @returns View into externally accessible memory or null view
    memory::DynMdView<void> getExternalView() const;

    /// Check whether this object is null
    friend bool operator==(std::nullptr_t, const MemoryDescriptorRef& ref);

    /// Check whether this object is null
    friend bool operator==(const MemoryDescriptorRef& ref, std::nullptr_t);

    /// Check whether this object is null
    friend bool operator!=(std::nullptr_t, const MemoryDescriptorRef& ref);

    /// Check whether this object is null
    friend bool operator!=(const MemoryDescriptorRef& ref, std::nullptr_t);
private:
    MemoryDescriptorView* m_parent = nullptr;
    memory::DynMdSpan m_span;
};

std::ostream& operator<<(std::ostream& os, const MemoryDescriptorRef& ref);

}

#endif // MEMORY_DESCRIPTOR_REF_HPP
