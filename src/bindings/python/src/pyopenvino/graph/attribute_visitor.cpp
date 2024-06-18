// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/attribute_visitor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"

namespace py = pybind11;

// void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;

//  void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override;

template <typename AT>
void visit_attribute(py::dict& attributes,
                     const std::pair<py::handle, py::handle>& attribute,
                     ov::AttributeVisitor* visitor) {
    auto attr_casted = attribute.second.cast<AT>();
    visitor->on_attribute<AT>(attribute.first.cast<std::string>(), attr_casted);
    attributes[attribute.first] = attr_casted;

    return;
};

void regclass_graph_AttributeVisitor(py::module m) {
    py::class_<ov::AttributeVisitor, std::shared_ptr<ov::AttributeVisitor>> visitor(m, "AttributeVisitor");

    visitor.def(
        "on_attributes",
        [](ov::AttributeVisitor* self, py::dict& attributes) {
            py::object float_32_type = py::module_::import("numpy").attr("float32");
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
                } else if (py::isinstance<py::array>(attribute.second)) {
                    // numpy arrays
                    auto _array = attribute.second.cast<py::array>();
                    if (py::isinstance<py::array_t<float>>(_array)) {
                        visit_attribute<std::vector<float>>(attributes, std::make_pair(attribute.first, _array), self);
                    } else {
                        OPENVINO_THROW("Unsupported NumPy array dtype.");
                    }
                } else if (py::isinstance<py::list>(attribute.second)) {
                    // python list
                    auto _list = attribute.second.cast<py::list>();

                    enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, DOUBLE, FLOAT };
                    PY_TYPE detected_type = PY_TYPE::UNKNOWN;

                    for (const auto& it : _list) {
                        auto check_type = [&](PY_TYPE type) {
                            if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                                detected_type = type;
                                return;
                            }
                            OPENVINO_THROW("Incorrect attribute. Mixed types in the list are not allowed.");
                        };
                        if (py::isinstance<py::str>(it)) {
                            check_type(PY_TYPE::STR);
                        } else if (py::isinstance<py::int_>(it)) {
                            check_type(PY_TYPE::INT);
                        } else if (py::isinstance<py::float_>(it)) {
                            check_type(PY_TYPE::DOUBLE);
                        } else if (py::isinstance(it, float_32_type)) {
                            check_type(PY_TYPE::FLOAT);
                        }
                    }

                    OPENVINO_ASSERT(!_list.empty(), "Attributes list is empty.");

                    switch (detected_type) {
                    case PY_TYPE::STR:
                        visit_attribute<std::vector<std::string>>(attributes,
                                                                  std::make_pair(attribute.first, _list),
                                                                  self);
                        break;
                    case PY_TYPE::DOUBLE:
                        // python float is float64 in C++
                        visit_attribute<std::vector<double>>(attributes, std::make_pair(attribute.first, _list), self);
                        break;
                    case PY_TYPE::FLOAT:
                        visit_attribute<std::vector<float>>(attributes, std::make_pair(attribute.first, _list), self);
                        break;
                    case PY_TYPE::INT:
                        visit_attribute<std::vector<int64_t>>(attributes, std::make_pair(attribute.first, _list), self);
                        break;
                    default:
                        OPENVINO_ASSERT(false, "Unsupported attribute type in provided list.");
                    }
                } else {
                    OPENVINO_ASSERT(false, "Unsupported attribute type.");
                }
            }
        },
        py::return_value_policy::reference_internal);
}
