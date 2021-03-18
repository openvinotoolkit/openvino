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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/variant.hpp"
#include "pyngraph/node.hpp"
#include "pyngraph/rt_map.hpp"
#include "pyngraph/variant.hpp"

namespace py = pybind11;

using PyRTMap = std::map<std::string, std::shared_ptr<ngraph::Variant>>;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_pyngraph_Node(py::module m)
{
    py::class_<ngraph::Node, std::shared_ptr<ngraph::Node>> node(m, "Node", py::dynamic_attr());
    node.doc() = "ngraph.impl.Node wraps ngraph::Node";
    node.def(
        "__add__",
        [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
            return std::make_shared<ngraph::op::v1::Add>(a, b);
        },
        py::is_operator());
    node.def(
        "__sub__",
        [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
            return std::make_shared<ngraph::op::v1::Subtract>(a, b);
        },
        py::is_operator());
    node.def(
        "__mul__",
        [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
            return std::make_shared<ngraph::op::v1::Multiply>(a, b);
        },
        py::is_operator());
    node.def(
        "__div__",
        [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
            return std::make_shared<ngraph::op::v1::Divide>(a, b);
        },
        py::is_operator());
    node.def(
        "__truediv__",
        [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
            return std::make_shared<ngraph::op::v1::Divide>(a, b);
        },
        py::is_operator());

    node.def("__repr__", [](const ngraph::Node& self) {
        std::string type_name = self.get_type_name();
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i)
        {
            if (i > 0)
            {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + type_name + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });

    node.def("get_element_type", &ngraph::Node::get_element_type);
    node.def("get_output_size", &ngraph::Node::get_output_size);
    node.def("get_output_element_type", &ngraph::Node::get_output_element_type);
    node.def("get_output_shape", &ngraph::Node::get_output_shape);
    node.def("get_output_partial_shape", &ngraph::Node::get_output_partial_shape);
    node.def("get_type_name", &ngraph::Node::get_type_name);
    node.def("get_name", &ngraph::Node::get_name);
    node.def("get_friendly_name", &ngraph::Node::get_friendly_name);
    node.def("set_friendly_name", &ngraph::Node::set_friendly_name);
    node.def("input", (ngraph::Input<ngraph::Node>(ngraph::Node::*)(size_t)) & ngraph::Node::input);
    node.def("inputs",
             (std::vector<ngraph::Input<ngraph::Node>>(ngraph::Node::*)()) & ngraph::Node::inputs);
    node.def("output",
             (ngraph::Output<ngraph::Node>(ngraph::Node::*)(size_t)) & ngraph::Node::output);
    node.def("outputs",
             (std::vector<ngraph::Output<ngraph::Node>>(ngraph::Node::*)()) &
                 ngraph::Node::outputs);
    node.def("get_rt_info",
             (PyRTMap & (ngraph::Node::*)()) & ngraph::Node::get_rt_info,
             py::return_value_policy::reference_internal);

    node.def_property_readonly("shape", &ngraph::Node::get_shape);
    node.def_property_readonly("name", &ngraph::Node::get_name);
    node.def_property_readonly("rt_info",
                               (PyRTMap & (ngraph::Node::*)()) & ngraph::Node::get_rt_info,
                               py::return_value_policy::reference_internal);
    node.def_property(
        "friendly_name", &ngraph::Node::get_friendly_name, &ngraph::Node::set_friendly_name);

    node.def("_get_attributes", [](const std::shared_ptr<ngraph::Node>& self) {
        util::DictAttributeSerializer dict_serializer(self);
        return dict_serializer.get_attributes();
    });
    node.def(
        "_set_attribute",
        [](std::shared_ptr<ngraph::Node>& self, const std::string& atr_name, py::object value) {
            py::dict attr_dict;
            attr_dict[atr_name.c_str()] = value;
            std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>> variables;
            util::DictAttributeDeserializer dict_deserializer(attr_dict, variables);
            self->visit_attributes(dict_deserializer);
        });
}
