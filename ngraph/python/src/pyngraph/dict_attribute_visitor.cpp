//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/opsets/opset6.hpp"

namespace py = pybind11;

util::DictAttributeDeserializer::DictAttributeDeserializer(
    const py::dict& attributes,
    std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& variables)
    : m_attributes(attributes)
    , m_variables(variables)
{
}

void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ngraph::ValueAccessor<void>& adapter)
{
    if (m_attributes.contains(name))
    {
        if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<
                std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>>>(
                &adapter))
        {
            std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>
                input_descs;
            const py::dict& input_desc = m_attributes[name.c_str()].cast<py::dict>();
            const auto& merged_input_desc = input_desc["merged_input_desc"].cast<py::list>();
            const auto& slice_input_desc = input_desc["slice_input_desc"].cast<py::list>();
            const auto& invariant_input_desc = input_desc["invariant_input_desc"].cast<py::list>();
            for (py::handle h : slice_input_desc)
            {
                const py::dict& desc = h.cast<py::dict>();
                auto slice_in =
                    std::make_shared<ngraph::op::util::SubGraphOp::SliceInputDescription>(
                        desc["input_idx"].cast<int64_t>(),
                        desc["body_parameter_idx"].cast<int64_t>(),
                        desc["start"].cast<int64_t>(),
                        desc["stride"].cast<int64_t>(),
                        desc["part_size"].cast<int64_t>(),
                        desc["end"].cast<int64_t>(),
                        desc["axis"].cast<int64_t>());
                input_descs.push_back(slice_in);
            }

            for (py::handle h : merged_input_desc)
            {
                const py::dict& desc = h.cast<py::dict>();
                auto merged_in =
                    std::make_shared<ngraph::op::util::SubGraphOp::MergedInputDescription>(
                        desc["input_idx"].cast<int64_t>(),
                        desc["body_parameter_idx"].cast<int64_t>(),
                        desc["body_value_idx"].cast<int64_t>());
                input_descs.push_back(merged_in);
            }

            for (py::handle h : invariant_input_desc)
            {
                const py::dict& desc = h.cast<py::dict>();
                auto invariant_in =
                    std::make_shared<ngraph::op::util::SubGraphOp::InvariantInputDescription>(
                        desc["input_idx"].cast<int64_t>(),
                        desc["body_parameter_idx"].cast<int64_t>());
                input_descs.push_back(invariant_in);
            }
            a->set(input_descs);
        }
        else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<
                     std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>>>(&adapter))
        {
            std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>
                output_descs;
            const py::dict& output_desc = m_attributes[name.c_str()].cast<py::dict>();
            const auto& body_output_desc = output_desc["body_output_desc"].cast<py::list>();
            const auto& concat_output_desc = output_desc["concat_output_desc"].cast<py::list>();
            for (py::handle h : body_output_desc)
            {
                const py::dict& desc = h.cast<py::dict>();
                auto body_output =
                    std::make_shared<ngraph::op::util::SubGraphOp::BodyOutputDescription>(
                        desc["body_value_idx"].cast<int64_t>(),
                        desc["output_idx"].cast<int64_t>(),
                        desc["iteration"].cast<int64_t>());
                output_descs.push_back(body_output);
            }

            for (py::handle h : concat_output_desc)
            {
                const py::dict& desc = h.cast<py::dict>();
                auto concat_output =
                    std::make_shared<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(
                        desc["body_value_idx"].cast<int64_t>(),
                        desc["output_idx"].cast<int64_t>(),
                        desc["start"].cast<int64_t>(),
                        desc["stride"].cast<int64_t>(),
                        desc["part_size"].cast<int64_t>(),
                        desc["end"].cast<int64_t>(),
                        desc["axis"].cast<int64_t>());
                output_descs.push_back(concat_output);
            }
            a->set(output_descs);
        }
        /*        else if (const auto& a = ngraph::as_type<
                             ngraph::AttributeAdapter<ngraph::opset6::Loop::SpecialBodyPorts>>(&adapter))
                {
                    ngraph::opset6::Loop::SpecialBodyPorts special_body_ports;
                    const py::dict& special_ports_dict =
           m_attributes[name.c_str()].cast<py::dict>();
                    special_body_ports.body_condition_output_idx =
                        special_ports_dict["body_condition_output_idx"].cast<int64_t>();
                    special_body_ports.current_iteration_input_idx =
                        special_ports_dict["current_iteration_input_idx"].cast<int64_t>();
                    a->set(special_body_ports);
                }*/
        else if (const auto& a =
                     ngraph::as_type<ngraph::AttributeAdapter<std::shared_ptr<ngraph::Variable>>>(
                         &adapter))
        {
            std::string variable_id = m_attributes[name.c_str()].cast<std::string>();
            if (!m_variables.count(variable_id))
            {
                m_variables[variable_id] = std::make_shared<ngraph::Variable>(ngraph::VariableInfo{
                    ngraph::PartialShape::dynamic(), ngraph::element::dynamic, variable_id});
            }
            a->set(m_variables[variable_id]);
        }
        else
        {
            NGRAPH_CHECK(
                false, "No AttributeVisitor support for accessing attribute named: ", name);
        }
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
