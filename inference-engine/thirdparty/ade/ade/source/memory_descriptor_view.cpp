// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "memory/memory_descriptor_view.hpp"

#include <vector>

#include <util/algorithm.hpp>
#include <util/range.hpp>
#include <util/zip_range.hpp>

#include "memory/memory_descriptor.hpp"

namespace ade
{

struct MemoryDescriptorView::Connector final
{
    // We use linear searches here because number of listeners and views usually
    // will be very small
    struct ListenerDesc final
    {
        MemoryDescriptorView* owner;
        std::vector<IMemoryDescriptorViewListener*> listeners;
    };

    std::vector<ListenerDesc> listeners;

    struct OwnerComparator final
    {
        const MemoryDescriptorView* owner;
        bool operator()(const ListenerDesc& desc) const
        {
            ASSERT(nullptr != owner);
            ASSERT(nullptr != desc.owner);
            return owner == desc.owner;
        }
    };

    void addListener(MemoryDescriptorView* view, IMemoryDescriptorViewListener* listener)
    {
        ASSERT(nullptr != view);
        ASSERT(nullptr != listener);
        ASSERT(!contains(view, listener));
        findDesc(view).listeners.push_back(listener);
    }

    void removeListener(MemoryDescriptorView* view, IMemoryDescriptorViewListener* listener)
    {
        ASSERT(nullptr != view);
        ASSERT(nullptr != listener);
        ASSERT(contains(view, listener));
        auto& desc = findDesc(view);
        util::unstable_erase(desc.listeners, util::find(desc.listeners, listener));
        ASSERT(!contains(view, listener));
    }

    void onDestroy(MemoryDescriptorView* view)
    {
        ASSERT(nullptr != view);
        auto it = util::find_if(listeners, OwnerComparator{view});
        if (listeners.end() != it)
        {
            for (auto& listener: it->listeners)
            {
                ASSERT(nullptr != listener);
                listener->destroy();
            }
            util::unstable_erase(listeners, it);
        }
    }

    bool contains(const MemoryDescriptorView* view, const IMemoryDescriptorViewListener* listener) const
    {
        ASSERT(nullptr != view);
        ASSERT(nullptr != listener);
        auto it = util::find_if(listeners, OwnerComparator{view});
        if (listeners.end() == it)
        {
            return false;
        }
        return it->listeners.end() != util::find(it->listeners, listener);
    }

    ListenerDesc& findDesc(MemoryDescriptorView* view)
    {
        ASSERT(nullptr != view);
        auto it = util::find_if(listeners, OwnerComparator{view});
        if(listeners.end() != it)
        {
            return *it;
        }
        listeners.push_back(ListenerDesc{view, {}});
        return listeners.back();
    }

    std::size_t listenersCount() const
    {
        std::size_t ret = 0;
        for (auto& desc: listeners)
        {
            ret += desc.listeners.size();
        }
        return ret;
    }

    ~Connector()
    {
        ASSERT(0 == listenersCount());
    }
};


MemoryDescriptorView::MemoryDescriptorView()
{

}

MemoryDescriptorView::MemoryDescriptorView(MemoryDescriptor& descriptor,
                                           const memory::DynMdSpan& span,
                                           RetargetableState retargetable):
    m_parent(&descriptor),
    m_span(span),
    m_retargetable(retargetable),
    m_connector(std::make_shared<Connector>())
{
    checkSpans(descriptor);
}

MemoryDescriptorView::MemoryDescriptorView(MemoryDescriptorView& parent,
                                           const memory::DynMdSpan& span):
    m_parent_view(&parent),
    m_span(span),
    m_retargetable(parent.retargetableState()),
    m_connector(parent.m_connector)
{

}

MemoryDescriptorView::~MemoryDescriptorView()
{
    if (nullptr != m_connector)
    {
        m_connector->onDestroy(this);
    }
}

void MemoryDescriptorView::retarget(MemoryDescriptor& newParent,
                                    const memory::DynMdSpan& newSpan)
{
    ASSERT(isRetargetable());
    ASSERT(nullptr != m_parent);
    ASSERT(nullptr == m_parent_view);
    ASSERT(nullptr != m_connector);
    const auto size = m_span.size();
    ASSERT(newSpan.size() == size);
    checkSpans(newParent);

    for (auto& desc: m_connector->listeners)
    {
        auto owner = desc.owner;
        ASSERT(nullptr != owner);
        for (auto listener: desc.listeners)
        {
            ASSERT(nullptr != listener);
            const auto origSpan = owner->span();
            const auto origin = origSpan.origin();
            const auto updatedSpan = util::make_span(origin, origSpan.size());
            if (owner == this)
            {
                ASSERT(updatedSpan == newSpan);
            }
            listener->retarget(*m_parent, origSpan, newParent, updatedSpan);
        }
    }

    m_span = newSpan;
    m_parent = &newParent;

    for (auto& desc: m_connector->listeners)
    {
        for (auto listener: desc.listeners)
        {
            ASSERT(nullptr != listener);
            listener->retargetComplete();
        }
    }
}

MemoryDescriptorView::RetargetableState MemoryDescriptorView::retargetableState() const
{
    return m_retargetable;
}

bool MemoryDescriptorView::isRetargetable() const
{
    return m_retargetable == Retargetable;
}

void MemoryDescriptorView::addListener(IMemoryDescriptorViewListener* listener)
{
    ASSERT(nullptr != listener);
    ASSERT(nullptr != m_connector);
    m_connector->addListener(this, listener);
}

void MemoryDescriptorView::removeListener(IMemoryDescriptorViewListener* listener)
{
    ASSERT(nullptr != listener);
    ASSERT(nullptr != m_connector);
    m_connector->removeListener(this, listener);
}

memory::DynMdSpan MemoryDescriptorView::span() const
{
    ASSERT(nullptr != *this);
    if (nullptr != m_parent_view)
    {
        return m_span + m_parent_view->span().origin();
    }
    return m_span;
}

memory::DynMdSize MemoryDescriptorView::size() const
{
    ASSERT(nullptr != *this);
    return m_span.size();
}

std::size_t MemoryDescriptorView::elementSize() const
{
    ASSERT(nullptr != getDescriptor());
    return getDescriptor()->elementSize();
}

MemoryDescriptor* MemoryDescriptorView::getDescriptor()
{
    if (nullptr != m_parent_view)
    {
        return m_parent_view->getDescriptor();
    }
    return m_parent;
}

const MemoryDescriptor* MemoryDescriptorView::getDescriptor() const
{
    if (nullptr != m_parent_view)
    {
        return m_parent_view->getDescriptor();
    }
    return m_parent;
}

MemoryDescriptorView* MemoryDescriptorView::getParentView()
{
    return m_parent_view;
}

const MemoryDescriptorView* MemoryDescriptorView::getParentView() const
{
    return m_parent_view;
}

memory::DynMdView<void> MemoryDescriptorView::getExternalView() const
{
    auto parent = getDescriptor();
    ASSERT(nullptr != parent);
    auto data = parent->getExternalView();
    if (nullptr == data)
    {
        return nullptr;
    }
    return data.slice(span());
}

MemoryDescriptorView::AccessHandle MemoryDescriptorView::access(const memory::DynMdSpan& span, MemoryAccessType accessType)
{
    ASSERT(nullptr != getDescriptor());
    return getDescriptor()->access(span + this->span().origin(), accessType);
}

void MemoryDescriptorView::commit(MemoryDescriptorView::AccessHandle handle)
{
    ASSERT(nullptr != getDescriptor());
    getDescriptor()->commit(handle);
}

bool operator==(std::nullptr_t, const MemoryDescriptorView& ref)
{
    return ref.getDescriptor() == nullptr;
}

bool operator==(const MemoryDescriptorView& ref, std::nullptr_t)
{
    return ref.getDescriptor() == nullptr;
}

bool operator!=(std::nullptr_t, const MemoryDescriptorView& ref)
{
    return ref.getDescriptor() != nullptr;
}

bool operator!=(const MemoryDescriptorView& ref, std::nullptr_t)
{
    return ref.getDescriptor() != nullptr;
}

void MemoryDescriptorView::checkSpans(MemoryDescriptor& descriptor) const
{
    ASSERT(descriptor.dimensions().dims_count() == m_span.dims_count());
    for (auto i: util::iota(m_span.dims_count()))
    {
        auto& val = m_span[i];
        ASSERT(val.begin >= 0);
        ASSERT(val.end <= descriptor.dimensions()[i]);
    }
}

void* getViewDataPtr(ade::MemoryDescriptorView& view, std::size_t offset)
{
    ASSERT(nullptr != view);
    auto data = view.getExternalView().mem;
    ASSERT(nullptr != data);
    const auto newSize = data.size - offset;
    ASSERT(newSize > 0);
    return data.Slice(offset, newSize).data;
}

void copyFromViewMemory(void* dst, ade::MemoryDescriptorView& view)
{
    ASSERT(nullptr != dst);
    ASSERT(nullptr != view);
    copyFromViewMemory(dst, view.getExternalView());
}

void copyToViewMemory(const void* src, ade::MemoryDescriptorView& view)
{
    ASSERT(nullptr != src);
    ASSERT(nullptr != view);
    copyToViewMemory(src, view.getExternalView());
}

void copyFromViewMemory(void* dst, ade::memory::DynMdView<void> view)
{
    ASSERT(nullptr != dst);
    ASSERT(nullptr != view);
    const auto size = view.sizeInBytes();
    util::raw_copy(view.mem, util::memory_range(dst, size));
}

void copyToViewMemory(const void* src, ade::memory::DynMdView<void> view)
{
    ASSERT(nullptr != src);
    ASSERT(nullptr != view);
    const auto size = view.sizeInBytes();
    util::raw_copy(util::memory_range(src, size), view.mem);
}

}
