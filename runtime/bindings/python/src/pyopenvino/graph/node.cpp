// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/variant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/graph/rt_map.hpp"
#include "pyopenvino/graph/variant.hpp"

class PyNode : public ov::Node {
public:
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ov::Node>, ov::Node, clone_with_new_inputs, inputs);
    }

    const type_info_t& get_type_info() const override {
        PYBIND11_OVERRIDE_PURE(type_info_t&, ov::Node, get_type_info, );
    }
};

namespace py = pybind11;

using PyRTMap = std::map<std::string, std::shared_ptr<ov::Variant>>;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_Node(py::module m) {
    py::class_<ov::Node, std::shared_ptr<ov::Node>, PyNode> node(m, "Node", py::dynamic_attr());
    node.doc() = "openvino.impl.Node wraps ov::Node";
    node.def(
        "__add__",
        [](const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node> b) {
            return std::make_shared<ov::op::v1::Add>(a, b);
        },
        py::is_operator());
    node.def(
        "__sub__",
        [](const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node> b) {
            return std::make_shared<ov::op::v1::Subtract>(a, b);
        },
        py::is_operator());
    node.def(
        "__mul__",
        [](const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node> b) {
            return std::make_shared<ov::op::v1::Multiply>(a, b);
        },
        py::is_operator());
    node.def(
        "__div__",
        [](const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node> b) {
            return std::make_shared<ov::op::v1::Divide>(a, b);
        },
        py::is_operator());
    node.def(
        "__truediv__",
        [](const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node> b) {
            return std::make_shared<ov::op::v1::Divide>(a, b);
        },
        py::is_operator());

    node.def("__repr__", [](const ov::Node& self) {
        std::string type_name = self.get_type_name();
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i) {
            if (i > 0) {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + type_name + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });

    node.def("get_element_type",
             &ov::Node::get_element_type,
             R"(
                Checks that there is exactly one output and returns it's element type.

                Returns
                ----------
                get_element_type : Type
                    Type of the output.
             )");
    node.def("get_output_size",
             &ov::Node::get_output_size,
             R"(
                Returns the number of outputs from the node.

                Returns
                ----------
                get_element_type : int
                    Number of outputs.
             )");
    node.def("get_output_element_type",
             &ov::Node::get_output_element_type,
             py::arg("i"),
             R"(
                Returns the element type for output i

                Parameters
                ----------
                i : int
                    Index of the output.

                Returns
                ----------
                get_output_element_type : Type
                    Type of the output i
             )");
    node.def("get_output_shape",
             &ov::Node::get_output_shape,
             py::arg("i"),
             R"(
                Returns the shape for output i

                Parameters
                ----------
                i : int
                    Index of the output.

                Returns
                ----------
                get_output_shape : Shape
                    Shape of the output i
             )");
    node.def("get_output_partial_shape",
             &ov::Node::get_output_partial_shape,
             py::arg("i"),
             R"(
                Returns the partial shape for output i

                Parameters
                ----------
                i : int
                    Index of the output.

                Returns
                ----------
                get_output_partial_shape : PartialShape
                    PartialShape of the output i
             )");
    node.def("get_type_name",
             &ov::Node::get_type_name,
             R"(
                Returns Type's name from the node.

                Returns
                ----------
                get_type_name : str
                    String repesenting Type's name. 
             )");
    node.def("get_name",
             &ov::Node::get_name,
             R"(
                Get the unique name of the node

                Returns
                ----------
                get_name : str
                    Unique name of the node.
             )");
    node.def("get_friendly_name",
             &ov::Node::get_friendly_name,
             R"(
                Gets the friendly name for a node. If no friendly name has 
                been set via set_friendly_name then the node's unique name
                is returned.

                Returns
                ----------
                get_name : str
                    Friendly name of the node.
             )");
    node.def("get_type_info", &ov::Node::get_type_info);
    node.def("set_friendly_name",
             &ov::Node::set_friendly_name,
             py::arg("name"),
             R"(
                Sets a friendly name for a node. This does not overwrite the unique name
                of the node and is retrieved via get_friendly_name(). Used mainly for 
                debugging. The friendly name may be set exactly once.

                Parameters
                ----------
                name : str
                    Friendly name to set.
             )");
    node.def("input",
             (ov::Input<ov::Node>(ov::Node::*)(size_t)) & ov::Node::input,
             py::arg("input_index"),
             R"(
                A handle to the input_index input of this node.

                Parameters
                ----------
                input_index : int
                    Index of Input.

                Returns
                ----------
                input : Input
                    Input of this node.
             )");
    node.def("inputs",
             (std::vector<ov::Input<ov::Node>>(ov::Node::*)()) & ov::Node::inputs,
             R"(
                A list containing a handle for each of this node's inputs, in order.

                Returns
                ----------
                inputs : List[Input]
                    List of node's inputs.
             )");
    node.def("output",
             (ov::Output<ov::Node>(ov::Node::*)(size_t)) & ov::Node::output,
             py::arg("output_index"),
             R"(
                A handle to the output_index output of this node.

                Parameters
                ----------
                output_index : int
                    Index of Output.

                Returns
                ----------
                input : Output
                    Output of this node.
             )");
    node.def("outputs",
             (std::vector<ov::Output<ov::Node>>(ov::Node::*)()) & ov::Node::outputs,
             R"(
                A list containing a handle for each of this node's outputs, in order.

                Returns
                ----------
                inputs : List[Output]
                    List of node's outputs.
             )");
    node.def("get_rt_info",
             (PyRTMap & (ov::Node::*)()) & ov::Node::get_rt_info,
             py::return_value_policy::reference_internal,
             R"(
                Returns PyRTMap which is a dictionary of user defined runtime info.

                Returns
                ----------
                get_rt_info : PyRTMap
                    A dictionary of user defined data.
             )");
    node.def("get_version",
             &ov::Node::get_version,
             R"(
                Returns operation's version of the node.

                Returns
                ----------
                get_version : int
                    Operation version.
             )");

    node.def("set_argument", &ov::Node::set_argument);
    node.def("set_arguments", [](const std::shared_ptr<ov::Node>& self, const ov::NodeVector& args) {
        self->set_arguments(args);
    });
    node.def("set_arguments", [](const std::shared_ptr<ov::Node>& self, const ov::OutputVector& args) {
        self->set_arguments(args);
    });

    node.def_property_readonly("shape", &ov::Node::get_shape);
    node.def_property_readonly("name", &ov::Node::get_name);
    node.def_property_readonly("rt_info",
                               (PyRTMap & (ov::Node::*)()) & ov::Node::get_rt_info,
                               py::return_value_policy::reference_internal);
    node.def_property_readonly("version", &ov::Node::get_version);
    node.def_property_readonly("type_info", &ov::Node::get_type_info);
    node.def_property("friendly_name", &ov::Node::get_friendly_name, &ov::Node::set_friendly_name);

    node.def("get_attributes", [](const std::shared_ptr<ov::Node>& self) {
        util::DictAttributeSerializer dict_serializer(self);
        return dict_serializer.get_attributes();
    });
    node.def("set_attribute", [](std::shared_ptr<ov::Node>& self, const std::string& atr_name, py::object value) {
        py::dict attr_dict;
        attr_dict[atr_name.c_str()] = value;
        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
        util::DictAttributeDeserializer dict_deserializer(attr_dict, variables);
        self->visit_attributes(dict_deserializer);
    });
    node.def("set_arguments", [](const std::shared_ptr<ov::Node>& self, const ov::OutputVector& arguments) {
        return self->set_arguments(arguments);
    });
    node.def("validate", [](const std::shared_ptr<ov::Node>& self) {
        return self->constructor_validate_and_infer_types();
    });
}
