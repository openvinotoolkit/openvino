// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MEMORY_DESCRIPTOR_VIEW_HPP
#define MEMORY_DESCRIPTOR_VIEW_HPP

#include <memory>

#include "memory/memory_types.hpp"
#include "memory/memory_accessor.hpp"

namespace ade
{
class MemoryDescriptor;

/// MemoryDescriptorView events listener
class IMemoryDescriptorViewListener
{
public:
    virtual ~IMemoryDescriptorViewListener() = default;

    /// This method is called when MemoryDescriptorView is about to be retargeted between
    /// different MemoryDescriptor's or between different parts of same descriptor
    ///
    /// @param oldDesc Previous descriptor
    /// @param oldSpan Previous span, always have size equal to span object initially created from
    /// @param newDes New descriptor
    /// @param newSpan New span, always have size equal to span object initially created from
    virtual void retarget(MemoryDescriptor& oldDesc, const memory::DynMdSpan& oldSpan,
                          MemoryDescriptor& newDesc, const memory::DynMdSpan& newSpan) = 0;

    /// This method is called immediately after successful retarget
    virtual void retargetComplete() = 0;

    /// This method is called when MemoryDescriptorView is about to be destroyed
    virtual void destroy() = 0;
};

/// View into MemoryDescriptor buffer
/// It can reference to a part of MemoryDescriptor buffer
/// It can be retargetable, e.g. dynamically switched between different MemoryDescriptor's
/// or different parts of MemoryDescriptor
class MemoryDescriptorView final
{
public:
    enum RetargetableState
    {
        Retargetable,
        NonRetargetable,
    };

    /// Contructs empty MemoryDescriptorView, whick cannot be retargeted
    MemoryDescriptorView();

    /// Constructs MemoryDescriptorView from descriptor and span
    ///
    /// @param descriptor Source descriptor
    /// @param span Span into descriptor, must be inside descriptor dimensions
    /// @param retargetable Is this view can be retargeted or not
    MemoryDescriptorView(MemoryDescriptor& descriptor,
                         const memory::DynMdSpan& span,
                         RetargetableState retargetable = NonRetargetable);

    /// Constructs MemoryDescriptorView from another MemoryDescriptorView and span
    /// Retargetable state will be inherited from parent
    ///
    /// @param parent Source view
    /// @param span Span into descriptor, must be inside parent dimensions
    MemoryDescriptorView(MemoryDescriptorView& parent,
                         const memory::DynMdSpan& span);

    ~MemoryDescriptorView();

    /// Retarged MemoryDescriptorView to a new descriptor
    /// MemoryDescriptorView must not be null and must be retargetable
    ///
    /// @param newParent New memory descriptor, can be same as previous
    /// @param newSpan New span must have same size as previous
    void retarget(MemoryDescriptor& newParent,
                  const memory::DynMdSpan& newSpan);

    /// Retrieve retargetable state
    ///
    /// @returns Retargetable state
    RetargetableState retargetableState() const;

    /// Check this object is retargetable
    ///
    /// @returns True if this object is retargetable
    bool isRetargetable() const;

    /// Add event listener
    ///
    /// @param listener Listener to be added, must not be null
    /// Same listener must not be added twice
    void addListener(IMemoryDescriptorViewListener* listener);

    /// Remove event listener
    ///
    /// @param listener Listener to be removed, must not be null
    /// Listener must be previously added via addListener call
    void removeListener(IMemoryDescriptorViewListener* listener);

    /// Get current span
    ///
    /// @returns Current span
    memory::DynMdSpan span() const;

    /// Get span size
    ///
    /// @returns size
    memory::DynMdSize size() const;

    /// Get buffer element size
    ///
    /// @returns Size in bytes
    std::size_t elementSize() const;

    /// Get current descriptor
    ///
    /// @returns Current descriptor
    MemoryDescriptor* getDescriptor();

    /// Get current descriptor
    ///
    /// @returns Current descriptor
    const MemoryDescriptor* getDescriptor() const;

    /// Get parent view
    ///
    /// @returns Pointer to parent MemoryDescriptorView or null
    MemoryDescriptorView* getParentView();

    /// Get parent view
    ///
    /// @returns Pointer to parent MemoryDescriptorView or null
    const MemoryDescriptorView* getParentView() const;

    /// Returns externally accessible memory view if any
    ///
    /// @returns View into externally accessible memory or null view
    memory::DynMdView<void> getExternalView() const;

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

    /// Check whether this object is null
    friend bool operator==(std::nullptr_t, const MemoryDescriptorView& ref);

    /// Check whether this object is null
    friend bool operator==(const MemoryDescriptorView& ref, std::nullptr_t);

    /// Check whether this object is null
    friend bool operator!=(std::nullptr_t, const MemoryDescriptorView& ref);

    /// Check whether this object is null
    friend bool operator!=(const MemoryDescriptorView& ref, std::nullptr_t);
private:
    void checkSpans(MemoryDescriptor& descriptor) const;

    MemoryDescriptor* m_parent = nullptr;
    MemoryDescriptorView* m_parent_view = nullptr;
    memory::DynMdSpan m_span;
    RetargetableState m_retargetable = NonRetargetable;

    struct Connector;
    std::shared_ptr<Connector> m_connector;
};

void* getViewDataPtr(MemoryDescriptorView& view, std::size_t offset = 0);
void copyFromViewMemory(void* dst, MemoryDescriptorView& view);
void copyToViewMemory(const void* src, MemoryDescriptorView& view);
void copyFromViewMemory(void* dst, memory::DynMdView<void> view);
void copyToViewMemory(const void* src, memory::DynMdView<void> view);
}

#endif // MEMORY_DESCRIPTOR_VIEW_HPP
