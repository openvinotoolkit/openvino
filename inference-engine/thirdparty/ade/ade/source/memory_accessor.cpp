// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "memory/memory_accessor.hpp"

#include <algorithm>

#include "util/zip_range.hpp"
#include "util/assert.hpp"

namespace ade
{

MemoryAccessor::MemoryAccessor()
{
}

MemoryAccessor::~MemoryAccessor()
{
    if (!m_activeHandles.empty())
    {
        onError("Data wasn't committed");
        abandonAllHandles();
    }

    ASSERT(m_activeHandles.empty() && "There are an opened handles");

    for (auto listener : m_accessListeners)
    {
        listener->memoryDescriptorDestroyedImpl();
    }
}

void MemoryAccessor::addListener(IMemoryAccessListener* listener)
{
    ASSERT(nullptr != listener);
    if (!m_activeHandles.empty())
    {
        onError("Data wasn't committed");
    }
    ASSERT(m_accessListeners.end() == std::find(m_accessListeners.begin(),
                                                m_accessListeners.end(),
                                                listener));
    m_accessListeners.push_back(listener);
}

void MemoryAccessor::removeListener(IMemoryAccessListener* listener)
{
    ASSERT(nullptr != listener);
    if (!m_activeHandles.empty())
    {
        onError("Data wasn't committed");
        abandonListenerHandles(listener);
    }
    auto it = std::find(m_accessListeners.begin(),
                        m_accessListeners.end(),
                        listener);
    ASSERT(m_accessListeners.end() != it);
    *it = m_accessListeners.back();
    m_accessListeners.pop_back();
}

MemoryAccessor::AccessHandle MemoryAccessor::access(const MemoryDescriptor& desc,
                                                    const memory::DynMdSpan& span,
                                                    MemoryAccessType accessType)
{
    ASSERT(nullptr != m_memory);
    return m_activeHandles.emplace(m_activeHandles.end(), this, desc, span, accessType);
}

void MemoryAccessor::commit(MemoryAccessor::AccessHandle handle)
{
    ASSERT(nullptr != m_memory);
    ASSERT(m_activeHandles.end() != handle);
    m_activeHandles.erase(handle);
}

void MemoryAccessor::setNewView(const memory::DynMdView<void>& mem)
{
    if (!m_activeHandles.empty())
    {
        onError("Data wasn't committed");
        abandonAllHandles();
    }
    for (auto& listener: m_accessListeners)
    {
        listener->memoryViewChangedImpl(m_memory, mem);
    }
    m_memory = mem;
}

void MemoryAccessor::abandonListenerHandles(IMemoryAccessListener* listener)
{
    ASSERT(nullptr != listener);

    for (auto& h: m_activeHandles)
    {
        h.abandon(listener);
    }
}

void MemoryAccessor::abandonAllHandles()
{
    for (auto& h: m_activeHandles)
    {
        h.abandon();
    }
    m_activeHandles.clear();
}

void MemoryAccessor::onError(const char* str)
{
    ASSERT(nullptr != str);
    if (m_errorListener)
    {
        m_errorListener(str);
    }
}

MemoryAccessor::SavedHandles::SavedHandles(MemoryAccessor* parent,
                                           const MemoryDescriptor& desc,
                                           const memory::DynMdSpan& span,
                                           MemoryAccessType accessType)
{
    ASSERT(nullptr != parent);
    using namespace util;

    // TODO: exception safety
    for (auto&& i: indexed(parent->m_accessListeners))
    {
        auto listener = value(i);
        if (0 == index(i))
        {
            handle = listener->access(desc, span, accessType);
        }
        else
        {
            handles.emplace_back(listener->access(desc, span, accessType));
        }
    }
}

MemoryAccessor::SavedHandles::~SavedHandles()
{

}

void MemoryAccessor::SavedHandles::abandon(IMemoryAccessListener* listener)
{
    ASSERT(nullptr != listener);

    if (listener == handle.get_deleter().listener)
    {
        handle.release();
    }
    for (auto& h: handles)
    {
        if (listener == h.get_deleter().listener)
        {
            h.release();
        }
    }
}

void MemoryAccessor::SavedHandles::abandon()
{
    handle.release();
    for (auto& h: handles)
    {
        h.release();
    }
}
}
