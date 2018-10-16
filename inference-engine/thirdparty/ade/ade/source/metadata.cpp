// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include "util/assert.hpp"

namespace ade
{

Metadata::Metadata()
{

}

util::any& Metadata::operator[](const MetadataId& id)
{
    ASSERT(nullptr != id);
    return m_data[id];
}

bool Metadata::contains(const MetadataId& id) const
{
    ASSERT(nullptr != id);
    return m_data.end() != m_data.find(id);
}

void Metadata::erase(const MetadataId& id)
{
    m_data.erase(id);
}

Metadata::MetadataRange Metadata::all()
{
    return util::toRange(m_data);
}

Metadata::MetadataCRange Metadata::all() const
{
    return util::toRange(m_data);
}

void Metadata::copyFrom(const Metadata& data)
{
    m_data = data.m_data;
}

std::size_t Metadata::IdHash::operator()(const MetadataId& id) const
{
    return std::hash<decltype(MetadataId::m_id)>()(id.m_id);
}

MetadataId::MetadataId(void* id):
    m_id(id)
{
    ASSERT(nullptr != m_id);
}

bool MetadataId::operator==(const MetadataId& other) const
{
    return m_id == other.m_id;
}

bool MetadataId::operator!=(const MetadataId& other) const
{
    return m_id != other.m_id;
}

bool MetadataId::isNull() const
{
    return nullptr == m_id;
}

bool operator==(std::nullptr_t, const MetadataId& other)
{
    return other.isNull();
}

bool operator==(const MetadataId& other, std::nullptr_t)
{
    return other.isNull();
}

bool operator!=(std::nullptr_t, const MetadataId& other)
{
    return !other.isNull();
}

bool operator!=(const MetadataId& other, std::nullptr_t)
{
    return !other.isNull();
}

}
