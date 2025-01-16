// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/attribute_visitor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

template <typename AT>
void visit_attribute(py::dict& attributes,
                     const std::pair<py::handle, py::handle>& attribute,
                     ov::AttributeVisitor* visitor) {
    auto attr_casted = attribute.second.cast<AT>();
    visitor->on_attribute<AT>(attribute.first.cast<std::string>(), attr_casted);
    attributes[attribute.first] = std::move(attr_casted);

    return;
};

void regclass_graph_AttributeVisitor(py::module m) {
    using PY_TYPE = Common::utils::PY_TYPE;

    py::class_<ov::AttributeVisitor, std::shared_ptr<ov::AttributeVisitor>> visitor(m, "AttributeVisitor");

    visitor.def(
        "on_attributes",
        [](ov::AttributeVisitor* self, py::dict& attributes) {
            py::object float_32_type = py::module_::import("numpy").attr("float32");
            py::object model = py::module_::import("openvino").attr("Model");
            for (const auto& attribute : attributes) {
                if (py::isinstance<py::bool_>(attribute.second)) {
                    visit_attribute<bool>(attributes, attribute, self);
                } else if (py::isinstance<py::str>(attribute.second)) {
                    visit_attribute<std::string>(attributes, attribute, self);
                } else if (py::isinstance<py::int_>(attribute.second)) {
                    visit_attribute<int64_t>(attributes, attribute, self);
                } else if (py::isinstance<py::float_>(attribute.second)) {
                    visit_attribute<double>(attributes, attribute, self);
                } else if (py::isinstance(attribute.second, float_32_type)) {
                    visit_attribute<float>(attributes, attribute, self);
                } else if (py::isinstance<ov::Model>(attribute.second)) {
                    visit_attribute<std::shared_ptr<ov::Model>>(attributes, attribute, self);
                } else if (py::isinstance(attribute.second, model)) {
                    auto attr_casted = attribute.second.attr("_Model__model").cast<std::shared_ptr<ov::Model>>();
                    self->on_attribute<std::shared_ptr<ov::Model>>(attribute.first.cast<std::string>(), attr_casted);
                    attributes[attribute.first] = std::move(attr_casted);
                } else if (py::isinstance<ov::Dimension>(attribute.second)) {
                    visit_attribute<ov::Dimension>(attributes, attribute, self);
                } else if (py::isinstance<ov::PartialShape>(attribute.second)) {
                    visit_attribute<ov::PartialShape>(attributes, attribute, self);
                } else if (py::isinstance<py::array>(attribute.second)) {
                    // numpy arrays
                    auto _array = attribute.second.cast<py::array>();
                    if (py::isinstance<py::array_t<float>>(_array)) {
                        visit_attribute<std::vector<float>>(attributes, std::make_pair(attribute.first, _array), self);
                    } else {
                        throw py::type_error("Unsupported NumPy array dtype: " + std::string(py::str(_array.dtype())));
                    }
                } else if (py::isinstance<py::list>(attribute.second)) {
                    // python list
                    auto _list = attribute.second.cast<py::list>();

                    OPENVINO_ASSERT(!_list.empty(), "Attributes list is empty.");

                    PY_TYPE detected_type = Common::utils::check_list_element_type(_list);

                    switch (detected_type) {
                    case PY_TYPE::STR:
                        visit_attribute<std::vector<std::string>>(attributes,
                                                                  std::make_pair(attribute.first, _list),
                                                                  self);
                        break;
                    case PY_TYPE::FLOAT:
                        // python float is float64 in C++
                        visit_attribute<std::vector<double>>(attributes, std::make_pair(attribute.first, _list), self);
                        break;
                    case PY_TYPE::INT:
                        visit_attribute<std::vector<int64_t>>(attributes, std::make_pair(attribute.first, _list), self);
                        break;
                    default:
                        throw py::type_error("Unsupported attribute type in provided list: " +
                                             std::string(py::str(_list[0].get_type())));
                    }
                } else {
                    throw py::type_error("Unsupported attribute type: " +
                                         std::string(py::str(attribute.second.get_type())));
                }
            }
        },
        py::return_value_policy::reference_internal);
}
