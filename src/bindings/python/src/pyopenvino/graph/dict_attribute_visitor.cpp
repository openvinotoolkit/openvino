// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// These are not used here, but needed in order to not violate ODR, since
// these are included in other translation units, and specialize some types.
// Related: https://github.com/pybind/pybind11/issues/1055
#include "dict_attribute_visitor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/op/loop.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace py = pybind11;

util::DictAttributeDeserializer::DictAttributeDeserializer(
    const py::dict& attributes,
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables)
    : m_attributes(attributes),
      m_variables(variables) {}

void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) {
    if (m_attributes.contains(name)) {
        if (const auto& a = ov::as_type<
                ov::AttributeAdapter<std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>>>(
                &adapter)) {
            std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>> input_descs;
            const py::dict& input_desc = m_attributes[name.c_str()].cast<py::dict>();
            if (input_desc.contains("slice_input_desc") && !input_desc["slice_input_desc"].is_none()) {
                for (py::handle h : input_desc["slice_input_desc"].cast<py::list>()) {
                    const py::dict& desc = h.cast<py::dict>();
                    auto slice_in = std::make_shared<ov::op::util::SubGraphOp::SliceInputDescription>(
                        desc["input_idx"].cast<int64_t>(),
                        desc["body_parameter_idx"].cast<int64_t>(),
                        desc["start"].cast<int64_t>(),
                        desc["stride"].cast<int64_t>(),
                        desc["part_size"].cast<int64_t>(),
                        desc["end"].cast<int64_t>(),
                        desc["axis"].cast<int64_t>());
                    input_descs.push_back(slice_in);
                }
            }
            if (input_desc.contains("merged_input_desc") && !input_desc["merged_input_desc"].is_none()) {
                for (py::handle h : input_desc["merged_input_desc"].cast<py::list>()) {
                    const py::dict& desc = h.cast<py::dict>();
                    auto merged_in = std::make_shared<ov::op::util::SubGraphOp::MergedInputDescription>(
                        desc["input_idx"].cast<int64_t>(),
                        desc["body_parameter_idx"].cast<int64_t>(),
                        desc["body_value_idx"].cast<int64_t>());
                    input_descs.push_back(merged_in);
                }
            }

            if (input_desc.contains("invariant_input_desc") && !input_desc["invariant_input_desc"].is_none()) {
                for (py::handle h : input_desc["invariant_input_desc"].cast<py::list>()) {
                    const py::dict& desc = h.cast<py::dict>();
                    auto invariant_in = std::make_shared<ov::op::util::SubGraphOp::InvariantInputDescription>(
                        desc["input_idx"].cast<int64_t>(),
                        desc["body_parameter_idx"].cast<int64_t>());
                    input_descs.push_back(invariant_in);
                }
            }
            a->set(input_descs);
        } else if (const auto& a = ov::as_type<
                       ov::AttributeAdapter<std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>>>>(
                       &adapter)) {
            std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>> output_descs;
            const py::dict& output_desc = m_attributes[name.c_str()].cast<py::dict>();
            if (output_desc.contains("body_output_desc") && !output_desc["body_output_desc"].is_none()) {
                for (py::handle h : output_desc["body_output_desc"].cast<py::list>()) {
                    const py::dict& desc = h.cast<py::dict>();
                    auto body_output = std::make_shared<ov::op::util::SubGraphOp::BodyOutputDescription>(
                        desc["body_value_idx"].cast<int64_t>(),
                        desc["output_idx"].cast<int64_t>(),
                        desc["iteration"].cast<int64_t>());
                    output_descs.push_back(body_output);
                }
            }
            if (output_desc.contains("concat_output_desc") && !output_desc["concat_output_desc"].is_none()) {
                for (py::handle h : output_desc["concat_output_desc"].cast<py::list>()) {
                    const py::dict& desc = h.cast<py::dict>();
                    auto concat_output = std::make_shared<ov::op::util::SubGraphOp::ConcatOutputDescription>(
                        desc["body_value_idx"].cast<int64_t>(),
                        desc["output_idx"].cast<int64_t>(),
                        desc["start"].cast<int64_t>(),
                        desc["stride"].cast<int64_t>(),
                        desc["part_size"].cast<int64_t>(),
                        desc["end"].cast<int64_t>(),
                        desc["axis"].cast<int64_t>());
                    output_descs.push_back(concat_output);
                }
            }
            a->set(output_descs);
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>>(&adapter)) {
            ov::op::v5::Loop::SpecialBodyPorts special_body_ports;
            const py::dict& special_ports_dict = m_attributes[name.c_str()].cast<py::dict>();
            special_body_ports.body_condition_output_idx =
                special_ports_dict["body_condition_output_idx"].cast<int64_t>();
            special_body_ports.current_iteration_input_idx =
                special_ports_dict["current_iteration_input_idx"].cast<int64_t>();
            a->set(special_body_ports);
        } else if (const auto& a =
                       ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
            std::string variable_id = m_attributes[name.c_str()].cast<std::string>();
            if (!m_variables.count(variable_id)) {
                m_variables[variable_id] = std::make_shared<ov::op::util::Variable>(
                    ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_id});
            }
            a->set(m_variables[variable_id]);
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            if (py::isinstance<ov::PartialShape>(m_attributes[name.c_str()])) {
                a->set(m_attributes[name.c_str()].cast<ov::PartialShape>());
            } else {
                OPENVINO_THROW("No AttributeVisitor support for accessing attribute named: ", name);
            }
        } else {
            OPENVINO_THROW("No AttributeVisitor support for accessing attribute named: ", name);
        }
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<bool>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) {
    if (m_attributes.contains(name)) {
        const auto& attribute = m_attributes[name.c_str()];
        if (py::isinstance<ov::element::Type>(attribute)) {
            adapter.set(attribute.cast<ov::element::Type>().get_type_name());
        } else {
            adapter.set(attribute.cast<std::string>());
        }
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<int8_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<int8_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<int16_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<int16_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<int32_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<int64_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<uint8_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<uint8_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<uint16_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<uint16_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<uint32_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<uint32_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<uint64_t>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<uint64_t>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<float>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<float>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<double>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<std::string>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<std::string>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<int8_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int8_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<int16_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int16_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<int32_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int32_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<int64_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<int64_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<uint8_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint8_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<uint16_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint16_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<uint32_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint32_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<uint64_t>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<uint64_t>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<float>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<float>>());
    }
}
void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::vector<double>>& adapter) {
    if (m_attributes.contains(name)) {
        adapter.set(m_attributes[name.c_str()].cast<std::vector<double>>());
    }
}

void util::DictAttributeDeserializer::on_adapter(const std::string& name,
                                                 ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) {
    if (m_attributes.contains(name)) {
        if (name == "body" || name == "then_body" || name == "else_body") {
            const py::dict& body_attrs = m_attributes[name.c_str()].cast<py::dict>();
            const auto& body_outputs = as_output_vector(body_attrs["results"].cast<ov::NodeVector>());
            const auto& body_parameters = body_attrs["parameters"].cast<ov::ParameterVector>();
            auto body = std::make_shared<ov::Model>(body_outputs, body_parameters);
            adapter.set(body);
        } else {
            OPENVINO_THROW("No AttributeVisitor support for accessing attribute named: ", name);
        }
    }
}

util::DictAttributeSerializer::DictAttributeSerializer(const std::shared_ptr<ov::Node>& node) {
    node->visit_attributes(*this);
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) {
    if (m_attributes.contains(name)) {
        OPENVINO_THROW("No AttributeVisitor support for accessing attribute named: ", name);
    }

    if (auto _adapter = dynamic_cast<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>*>(&adapter)) {
        m_attributes[name.c_str()] = _adapter->get()->get_info().variable_id;
    } else if (auto _adapter = dynamic_cast<ov::AttributeAdapter<ov::PartialShape>*>(&adapter)) {
        auto partial_shape = _adapter->get();
        std::vector<ov::Dimension::value_type> shape;
        for (const auto& dim : partial_shape) {
            shape.push_back(dim.is_dynamic() ? -1 : dim.get_length());
        }
        m_attributes[name.c_str()] = shape;
    }
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<int8_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<int16_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<uint8_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<uint16_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<uint32_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<uint64_t>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<float>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<std::string>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<int8_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<int16_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<int32_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<int64_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<uint8_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<uint16_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<uint32_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<uint64_t>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<float>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
void util::DictAttributeSerializer::on_adapter(const std::string& name,
                                               ov::ValueAccessor<std::vector<double>>& adapter) {
    m_attributes[name.c_str()] = adapter.get();
}
