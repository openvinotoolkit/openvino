// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ostream>

#include "memory/memory_descriptor_ref.hpp"

#include "memory/memory_descriptor_view.hpp"
#include "memory/memory_descriptor.hpp"

#include "util/md_io.hpp"

#include <util/iota_range.hpp>

namespace ade
{

MemoryDescriptorRef::MemoryDescriptorRef()
{

}

MemoryDescriptorRef::MemoryDescriptorRef(MemoryDescriptorView& view):
    m_parent(&view)
{
    m_span.redim(view.span().dims_count());
    for (auto i: util::iota(m_span.dims_count()))
    {
        m_span[i] = util::Span(0, view.span()[i].length());
    }
}

MemoryDescriptorRef::MemoryDescriptorRef(MemoryDescriptorView& view,
                                         const memory::DynMdSpan& span):
    m_parent(&view),
    m_span(span)
{

}

MemoryDescriptorRef::~MemoryDescriptorRef()
{

}

MemoryDescriptorView* MemoryDescriptorRef::getView()
{
    return m_parent;
}

const MemoryDescriptorView* MemoryDescriptorRef::getView() const
{
    return m_parent;
}

MemoryDescriptor*MemoryDescriptorRef::getDescriptor()
{
    if (nullptr == m_parent)
    {
        return nullptr;
    }
    return m_parent->getDescriptor();
}

const MemoryDescriptor* MemoryDescriptorRef::getDescriptor() const
{
    if (nullptr == m_parent)
    {
        return nullptr;
    }
    return m_parent->getDescriptor();
}

const memory::DynMdSpan& MemoryDescriptorRef::span() const
{
    ASSERT(nullptr != *this);
    return m_span;
}

memory::DynMdSize MemoryDescriptorRef::size() const
{
    return span().size();
}

std::size_t MemoryDescriptorRef::elementSize() const
{
    ASSERT(nullptr != getDescriptor());
    return getDescriptor()->elementSize();
}

memory::DynMdSpan MemoryDescriptorRef::originSpan() const
{
    ASSERT(nullptr != *this);
    return m_span + m_parent->span().origin();
}

memory::DynMdView<void> MemoryDescriptorRef::getExternalView() const
{
    ASSERT(nullptr != getDescriptor());
    auto srcView = getDescriptor()->getExternalView();
    if (nullptr == srcView)
    {
        return nullptr;
    }
    return srcView.slice(originSpan());
}

bool operator==(std::nullptr_t, const MemoryDescriptorRef& ref)
{
    return ref.m_parent == nullptr;
}

bool operator==(const MemoryDescriptorRef& ref, std::nullptr_t)
{
    return ref.m_parent == nullptr;
}

bool operator!=(std::nullptr_t, const MemoryDescriptorRef& ref)
{
    return ref.m_parent != nullptr;
}

bool operator!=(const MemoryDescriptorRef& ref, std::nullptr_t)
{
    return ref.m_parent != nullptr;
}

std::ostream& operator<<(std::ostream& os, const MemoryDescriptorRef& ref)
{
    if (nullptr == ref)
    {
        os << static_cast<void*>(nullptr);
    }
    else
    {
        os << "{";
        os << "span: " << ref.span() << ", ";
        if (ref.originSpan() != ref.span())
        {
            os << std::endl << "origin span: " << ref.originSpan() << ", ";
        }
        os << std::endl << "view: " << ref.getView() << " (" << ref.getView()->span() << "), ";
        os << std::endl << "descriptor: " << ref.getDescriptor() << " (" << ref.getDescriptor()->dimensions() << ") ";
        os << "}";
    }
    return os;
}

}
