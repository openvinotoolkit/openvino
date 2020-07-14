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

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "pyngraph/node.hpp"

namespace py = pybind11;

void regclass_pyngraph_Node(py::module m)
{
    py::class_<ngraph::Node, std::shared_ptr<ngraph::Node>> node(m, "Node", py::dynamic_attr());
    node.doc() = "ngraph.impl.Node wraps ngraph::Node";
    node.def("__add__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a + b;
             },
             py::is_operator());
    node.def("__sub__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a - b;
             },
             py::is_operator());
    node.def("__mul__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a * b;
             },
             py::is_operator());
    node.def("__div__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a / b;
             },
             py::is_operator());
    node.def("__truediv__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a / b;
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
    node.def("get_unique_name", &ngraph::Node::get_name);

    node.def_property("name", &ngraph::Node::get_friendly_name, &ngraph::Node::set_friendly_name);
    node.def_property_readonly("shape", &ngraph::Node::get_shape);

    node.def("_get_attributes", [](const std::shared_ptr<ngraph::Node>& self) {
        util::DictAttributeSerializer dict_serializer(self);
        return dict_serializer.get_attributes();
    });
    node.def(
        "_set_attribute",
        [](std::shared_ptr<ngraph::Node>& self, const std::string& atr_name, py::object value) {
            py::dict attr_dict;
            attr_dict[atr_name.c_str()] = value;
            util::DictAttributeDeserializer dict_deserializer(attr_dict);
            self->visit_attributes(dict_deserializer);
        });
}
