// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <regex>
#include <string>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/graph/rt_map.hpp"
#include "pyopenvino/utils/utils.hpp"

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

using PyRTMap = ov::Node::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_Node(py::module m) {
    py::class_<ov::Node, std::shared_ptr<ov::Node>, PyNode> node(m, "Node", py::dynamic_attr());
    node.doc() = "openvino.runtime.Node wraps ov::Node";
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

    // This function replaces NodeFactory's mechanism previously getting attributes of Nodes.
    // Python will call this method whenever requested attribute hasn't already been defined,
    // so all other properties like shape or name are prioritized from bindings itself.
    // TODO: is it possible to append these attributes to the instance of Node itself
    //       directly from pybind without returning class itself everytime?
    node.def("__getattr__", [](const std::shared_ptr<ov::Node>& self, const std::string& name) {
        // TODO: is it possible to cache serializer and regex?
        util::DictAttributeSerializer dict_serializer(self);

        // Look if there is a "get/set_*" pattern at the start of a string.
        // It means attribute was called as a function.
        std::smatch regex_match;
        std::regex look_for("get_|set_");

        if (std::regex_search(name, regex_match, look_for, std::regex_constants::match_continuous)) {
            // Strip prefix from the name and perform operation.
            auto stripped_name = regex_match.suffix().str();
            if (regex_match.str() == "get_") {
                if (dict_serializer.contains_attribute(stripped_name)) {
                    return py::cpp_function([self, stripped_name]() {
                        util::DictAttributeSerializer dict_serializer(self);
                        return dict_serializer.get_attribute<py::object>(stripped_name);
                    });
                }
                // Throw error with original name if stripped set_attribute was not found:
                throw py::attribute_error("'openvino.runtime.Node' object has no attribute '" + name + "'");
            } else {  // regex_match is equal to "set_"
                if (dict_serializer.contains_attribute(stripped_name)) {
                    return py::cpp_function([self, stripped_name](py::object& value) {
                        py::dict attr_dict;
                        attr_dict[stripped_name.c_str()] = value;
                        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
                        util::DictAttributeDeserializer dict_deserializer(attr_dict, variables);
                        self->visit_attributes(dict_deserializer);
                        return;
                    });
                }
                // Throw error with original name if stripped set_attribute was not found:
                throw py::attribute_error("'openvino.runtime.Node' object has no attribute '" + name + "'");
            }
        }

        // If nothing was found raise AttributeError:
        throw py::attribute_error("'openvino.runtime.Node' object has no attribute '" + name + "'");
    });

    node.def(
        "evaluate",
        [](const ov::Node& self,
           ov::TensorVector& output_values,
           const ov::TensorVector& input_values,
           const ov::EvaluationContext& evaluationContext) -> bool {
            return self.evaluate(output_values, input_values, evaluationContext);
        },
        py::arg("output_values"),
        py::arg("input_values"),
        py::arg("evaluationContext"),
        R"(
                Evaluate the node on inputs, putting results in outputs
                
                :param output_tensors: Tensors for the outputs to compute. One for each result.
                :type output_tensors: List[openvino.runtime.Tensor]
                :param input_tensors: Tensors for the inputs. One for each inputs.
                :type input_tensors: List[openvino.runtime.Tensor]
                :param evaluation_context: Storage of additional settings and attributes that can be used
                when evaluating the function. This additional information can be shared across nodes.
                :type evaluation_context: openvino.runtime.RTMap
                :rtype: bool
            )");
    node.def(
        "evaluate",
        [](const ov::Node& self, ov::TensorVector& output_values, const ov::TensorVector& input_values) -> bool {
            return self.evaluate(output_values, input_values);
        },
        py::arg("output_values"),
        py::arg("input_values"),
        R"(
                Evaluate the function on inputs, putting results in outputs

                :param output_tensors: Tensors for the outputs to compute. One for each result.
                :type output_tensors: List[openvino.runtime.Tensor]
                :param input_tensors: Tensors for the inputs. One for each inputs.
                :type input_tensors: List[openvino.runtime.Tensor]
                :rtype: bool
             )");
    node.def("get_instance_id",
             &ov::Node::get_instance_id,
             R"(
                Returns id of the node.
                May be used to compare nodes if they are same instances.

                :return: id of the node.
                :rtype: int
            )");

    node.def("get_input_tensor",
             &ov::Node::get_input_tensor,
             py::arg("index"),
             py::return_value_policy::reference_internal,
             R"(
                Returns the tensor for the node's input with index i

                :param index: Index of Input.
                :type index: int
                :return: Tensor of the input index
                :rtype: openvino._pyopenvino.DescriptorTensor
             )");
    node.def("get_element_type",
             &ov::Node::get_element_type,
             R"(
                Checks that there is exactly one output and returns it's element type.

                :return: Type of the output.
                :rtype: openvino.runtime.Type
             )");
    node.def("input_values",
             &ov::Node::input_values,
             R"(
                 Returns list of node's inputs, in order.

                 :return: List of node's inputs
                 :rtype: List[openvino.runtime.Input]
             )");
    node.def("input_value",
             &ov::Node::input_value,
             py::arg("index"),
             R"(
                Returns input of the node with index i

                :param index: Index of Input.
                :type index: int
                :return: Input of this node.
                :rtype: openvino.runtime.Input
             )");
    node.def("get_input_size",
             &ov::Node::get_input_size,
             R"(
                Returns the number of inputs to the node.

                :return: Number of inputs.
                :rtype: int
             )");
    node.def("get_input_element_type",
             &ov::Node::get_input_element_type,
             py::arg("index"),
             R"(
                Returns the element type for input index

                :param index: Index of the input.
                :type index: int
                :return: Type of the input index
                :rtype: openvino.Type
             )");
    node.def("get_input_partial_shape",
             &ov::Node::get_input_partial_shape,
             py::arg("index"),
             R"(
                Returns the partial shape for input index

                :param index: Index of the input.
                :type index: int
                :return: PartialShape of the input index
                :rtype: openvino.PartialShape
             )");
    node.def("get_input_shape",
             &ov::Node::get_input_shape,
             py::arg("index"),
             R"(
                Returns the shape for input index

                :param index: Index of the input.
                :type index: int
                :return: Shape of the input index
                :rtype: openvino.Shape
             )");
    node.def("set_output_type",
             &ov::Node::set_output_type,
             py::arg("index"),
             py::arg("element_type"),
             py::arg("shape"),
             R"(
                Sets output's element type and shape.

                :param index: Index of the output.
                :type index: int
                :param element_type: Element type of the output.
                :type element_type: openvino.Type
                :param shape: Shape of the output.
                :type shape: openvino.PartialShape
             )");
    node.def("set_output_size",
             &ov::Node::set_output_size,
             py::arg("size"),
             R"(
                Sets the number of outputs

                :param size: number of outputs.
                :type size: int
             )");
    node.def("get_output_size",
             &ov::Node::get_output_size,
             R"(
                Returns the number of outputs from the node.

                :return: Number of outputs.
                :rtype: int
             )");
    node.def("get_output_element_type",
             &ov::Node::get_output_element_type,
             py::arg("index"),
             R"(
                Returns the element type for output index

                :param index: Index of the output.
                :type index: int
                :return: Type of the output index
                :rtype: openvino.runtime.Type
             )");
    node.def("get_output_shape",
             &ov::Node::get_output_shape,
             py::arg("index"),
             R"(
                Returns the shape for output index

                :param index: Index of the output.
                :type index: int
                :return: Shape of the output index
                :rtype: openvino.runtime.Shape
             )");
    node.def("get_output_partial_shape",
             &ov::Node::get_output_partial_shape,
             py::arg("index"),
             R"(
                Returns the partial shape for output index

                :param index: Index of the output.
                :type index: int
                :return: PartialShape of the output index
                :rtype: openvino.runtime.PartialShape
             )");
    node.def("get_output_tensor",
             &ov::Node::get_output_tensor,
             py::arg("index"),
             py::return_value_policy::reference_internal,
             R"(
                Returns the tensor for output index

                :param index: Index of the output.
                :type index: int
                :return: Tensor of the output index
                :rtype: openvino._pyopenvino.DescriptorTensor
             )");
    node.def("get_type_name",
             &ov::Node::get_type_name,
             R"(
                Returns Type's name from the node.

                :return: String representing Type's name.
                :rtype: str
             )");
    node.def("get_name",
             &ov::Node::get_name,
             R"(
                Get the unique name of the node

                :return: Unique name of the node.
                :rtype: str
             )");
    node.def("get_friendly_name",
             &ov::Node::get_friendly_name,
             R"(
                Gets the friendly name for a node. If no friendly name has
                been set via set_friendly_name then the node's unique name
                is returned.

                :return: Friendly name of the node.
                :rtype: str
             )");
    node.def("get_type_info", &ov::Node::get_type_info);
    node.def("set_friendly_name",
             &ov::Node::set_friendly_name,
             py::arg("name"),
             R"(
                Sets a friendly name for a node. This does not overwrite the unique name
                of the node and is retrieved via get_friendly_name(). Used mainly for
                debugging. The friendly name may be set exactly once.

                :param name: Friendly name to set.
                :type name: str
             )");
    node.def("input",
             (ov::Input<ov::Node>(ov::Node::*)(size_t)) & ov::Node::input,
             py::arg("input_index"),
             R"(
                A handle to the input_index input of this node.

                :param input_index: Index of Input.
                :type input_index: int
                :return: Input of this node.
                :rtype: openvino.runtime.Input
             )");
    node.def("inputs",
             (std::vector<ov::Input<ov::Node>>(ov::Node::*)()) & ov::Node::inputs,
             R"(
                A list containing a handle for each of this node's inputs, in order.

                :return: List of node's inputs.
                :rtype: List[openvino.runtime.Input]
             )");
    node.def("output",
             (ov::Output<ov::Node>(ov::Node::*)(size_t)) & ov::Node::output,
             py::arg("output_index"),
             R"(
                A handle to the output_index output of this node.

                :param output_index: Index of Output.
                :type output_index: int
                :return: Output of this node.
                :rtype: openvino.runtime.Output
             )");
    node.def("outputs",
             (std::vector<ov::Output<ov::Node>>(ov::Node::*)()) & ov::Node::outputs,
             R"(
                A list containing a handle for each of this node's outputs, in order.

                :return: List of node's outputs.
                :rtype: List[openvino.runtime.Output]
             )");
    node.def("get_rt_info",
             (PyRTMap & (ov::Node::*)()) & ov::Node::get_rt_info,
             py::return_value_policy::reference_internal,
             R"(
                Returns PyRTMap which is a dictionary of user defined runtime info.

                :return: A dictionary of user defined data.
                :rtype: openvino.runtime.RTMap
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
    node.def("constructor_validate_and_infer_types", [](const std::shared_ptr<ov::Node>& self) {
        return self->constructor_validate_and_infer_types();
    });
    node.def(
        "validate_and_infer_types",
        [](const std::shared_ptr<ov::Node>& self) {
            return self->validate_and_infer_types();
        },
        R"(
        Verifies that attributes and inputs are consistent and computes output shapes and element types.
        Must be implemented by concrete child classes so that it can be run any number of times.
        
        Throws if the node is invalid.
    )");
}
