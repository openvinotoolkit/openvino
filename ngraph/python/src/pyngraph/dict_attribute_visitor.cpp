//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// These are not used here, but needed in order to not violate ODR, since
// these are included in other translation units, and specialize some types.
// Related: https://github.com/pybind/pybind11/issues/1055
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"

namespace py = pybind11;

util::DictAttributeDeserializer::DictAttributeDeserializer(const py::dict& attributes)
    : m_attributes(attributes)
{
}

void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<void>& adapter)
{
    if (m_attributes.contains(name))
    {
        NGRAPH_CHECK(false, "No AttributeVisitor support for accessing attribute named: ", name);
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<bool>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<bool>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<std::string>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::string>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<int8_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<int8_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<int16_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<int16_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<int32_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<int32_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<int64_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<int64_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<uint8_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<uint8_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<uint16_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<uint16_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<uint32_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<uint32_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<uint64_t>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<uint64_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<float>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<float>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<double>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<double>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<std::string>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<int8_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int8_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<int16_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int16_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int32_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int64_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint8_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint8_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint16_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint16_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint32_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint32_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint64_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<std::vector<float>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<float>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<double>>& adapter)
{
    if (m_attributes.contains(name))
    {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<double>>());
    }
}

util::DictAttributeSerializer::DictAttributeSerializer(const std::shared_ptr<ngraph::Node>& node)
{
    node->visit_attributes(*this);
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<void>& adapter)
{
    if (m_attributes.contains(name))
    {
        NGRAPH_CHECK(false, "No AttributeVisitor support for accessing attribute named: ", name);
    }
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<bool>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::string>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<int8_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<int16_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<int32_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<int64_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<uint8_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<uint16_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<uint32_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<uint64_t>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<float>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<double>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<int8_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<int16_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<int32_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<int64_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<uint8_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint16_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint32_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<float>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ngraph::ValueAccessor<std::vector<double>>& adapter)
{
    m_attributes[name.c_str()] = adapter.get();
}
