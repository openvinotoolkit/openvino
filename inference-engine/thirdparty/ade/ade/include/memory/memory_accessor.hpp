// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef ADE_MEMORYACCESSOR_HPP
#define ADE_MEMORYACCESSOR_HPP

#include <vector>
#include <list>
#include <functional>

#include "memory/memory_types.hpp"

#include "memory/memory_access_listener.hpp"

namespace ade
{

/// This class is used to notify listeners about externally accessible
/// data state changes
class MemoryAccessor final
{
    struct SavedHandles;
public:
    MemoryAccessor();
    MemoryAccessor(const MemoryAccessor&) = delete;
    MemoryAccessor& operator=(const MemoryAccessor&) = delete;
    ~MemoryAccessor();

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

    using AccessHandle = std::list<SavedHandles>::iterator;

    /// Notify all listeners about memory access
    ///
    /// @param desc The memory descriptor to be accessed
    /// @param span Span that represents a ROI of current view to be accessed
    /// @param accessType Access type for this view, it must be Read, Write or ReadWrite
    /// @returns Handle which later will be passed to commit
    AccessHandle access(const MemoryDescriptor& desc, const memory::DynMdSpan& span, MemoryAccessType accessType);

    /// Notify all listeners about memory commit
    ///
    /// @param handle Handle returned from successuful access call
    void commit(AccessHandle handle);

    /// Update memory view and notify all listeners
    ///
    /// @param mem New memory view
    void setNewView(const memory::DynMdView<void>& mem);

    /// Set MemoryAccessor usage errors listener
    ///
    /// @param listener Errors listener with signature void(const char*)
    template<typename F>
    void setErrorListener(F&& listener)
    {
        m_errorListener = std::forward<F>(listener);
    }

private:
    std::vector<IMemoryAccessListener*> m_accessListeners;

    memory::DynMdView<void> m_memory;

    struct SavedHandles
    {
        SavedHandles(MemoryAccessor* parent,
                     const MemoryDescriptor& desc,
                     const memory::DynMdSpan& span,
                     MemoryAccessType accessType);
        ~SavedHandles();
        SavedHandles(const SavedHandles&) = delete;
        SavedHandles& operator=(const SavedHandles&) = delete;
        SavedHandles(SavedHandles&&) = default;
        SavedHandles& operator=(SavedHandles&&) = default;

        // optimization to reduce heap allocation for common case
        IMemoryAccessListener::AccessHandlePtr handle;
        std::vector<IMemoryAccessListener::AccessHandlePtr> handles;

        void abandon(IMemoryAccessListener* listener);
        void abandon();
    };

    std::list<SavedHandles> m_activeHandles;

    std::function<void(const char*)> m_errorListener;

    void abandonListenerHandles(IMemoryAccessListener* listener);
    void abandonAllHandles();
    void onError(const char* str);
};

}

#endif // ADE_MEMORYACCESSOR_HPP
