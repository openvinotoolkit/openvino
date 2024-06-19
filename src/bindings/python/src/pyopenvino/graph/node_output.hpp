// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "openvino/core/node_output.hpp"
#include "pyopenvino/core/common.hpp"

#include <type_traits>

namespace py = pybind11;

using PyRTMap = ov::Node::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

// this function is overloaded in the corresponding cpp file with T=ov::Node
// it exposes additional functions with T = ov::Node, which are undefined with T = const ov::Node
template<typename T>
void def_type_dependent_functions(py::class_<ov::Output<T>, std::shared_ptr<ov::Output<T>>>& output);

template<>
void def_type_dependent_functions<ov::Node>(py::class_<ov::Output<ov::Node>,
                                            std::shared_ptr<ov::Output<ov::Node>>>& output);

template<>
void def_type_dependent_functions<const ov::Node>(py::class_<ov::Output<const ov::Node>,
                                                  std::shared_ptr<ov::Output<const ov::Node>>>& output);

template <typename VT>
void regclass_graph_Output(py::module m, std::string typestring)
{
    auto pyclass_name = py::detail::c_str((typestring + std::string("Output")));
    auto docs = py::detail::c_str(std::string("openvino.runtime.") + typestring + std::string("Output represents port/node output."));
    py::class_<ov::Output<VT>, std::shared_ptr<ov::Output<VT>>> output(m,
                                                                       pyclass_name,
                                                                       py::dynamic_attr());
    output.doc() = docs;

    // operator overloading
    output.def(py::self < py::self);
    output.def(py::self <= py::self);
    output.def(py::self > py::self);
    output.def(py::self >= py::self);
    output.def(py::self == py::self);
    output.def(py::self != py::self);

    output.def("__hash__", [](ov::Output<VT>& self) {
        return std::hash<VT*>()(self.get_node()) + self.get_index();
    });

    output.def("__copy__", [](ov::Output<VT>& self) {
        return ov::Output<VT>(self);
    });

    output.def("__deepcopy__", [typestring](ov::Output<VT>& self, py::dict& memo) {
        throw py::type_error("Cannot deepcopy 'openvino.runtime." + typestring + "Output' object.");
    });

    output.def("get_node",
               &ov::Output<VT>::get_node_shared_ptr,
               R"(
                Get node referenced by this output handle.

                :return: Node object referenced by this output handle.
                :rtype: openvino.runtime.Node
               )");
    output.def("get_index",
               &ov::Output<VT>::get_index,
               R"(
                The index of the output referred to by this output handle.

                :return: Index value as integer.
                :rtype: int
               )");
    output.def("get_any_name",
               &ov::Output<VT>::get_any_name,
               R"(
                One of the tensor names associated with this output.
                Note: first name in lexicographical order.

                :return: Tensor name as string.
                :rtype: str
               )");
    output.def("get_names",
               &ov::Output<VT>::get_names,
               R"(
                The tensor names associated with this output.

                :return: Set of tensor names.
                :rtype: Set[str]
               )");
    output.def("get_element_type",
               &ov::Output<VT>::get_element_type,
               R"(
                The element type of the output referred to by this output handle.

                :return: Type of the output.
                :rtype: openvino.runtime.Type
               )");
    output.def("get_shape",
               &ov::Output<VT>::get_shape,
               py::return_value_policy::copy,
               R"(
                The shape of the output referred to by this output handle.

                :return: Copy of Shape of the output.
                :rtype: openvino.runtime.Shape
               )");
    output.def("get_partial_shape",
               &ov::Output<VT>::get_partial_shape,
               py::return_value_policy::copy,
               R"(
                The partial shape of the output referred to by this output handle.

                :return: Copy of PartialShape of the output.
                :rtype: openvino.runtime.PartialShape
               )");
    output.def("get_target_inputs",
               &ov::Output<VT>::get_target_inputs,
               R"(
                A set containing handles for all inputs, targeted by the output,
                referenced by this output handle.

                :return: Set of Inputs.
                :rtype: Set[openvino.runtime.Input]
               )");
    output.def("_from_node", [](const std::shared_ptr<ov::Node>& node) {
               return ov::Output<ov::Node>(node);
               });
    output.def("get_tensor",
               &ov::Output<VT>::get_tensor,
               py::return_value_policy::reference_internal,
               R"(
                A reference to the tensor descriptor for this output.

                :return: Tensor of the output.
                :rtype: openvino._pyopenvino.DescriptorTensor
               )");
    output.def("get_rt_info",
             (ov::RTMap & (ov::Output<VT>::*)()) &  ov::Output<VT>::get_rt_info,
             py::return_value_policy::reference_internal,
             R"(
                Returns RTMap which is a dictionary of user defined runtime info.

                :return: A dictionary of user defined data.
                :rtype: openvino.runtime.RTMap
             )");
    output.def("__repr__", [](const ov::Output<VT>& self) {
        std::stringstream shape_type_ss;

        auto names_str = Common::docs::container_to_string(self.get_names(), ", ");
        shape_type_ss << " shape" << self.get_partial_shape() << " type: " << self.get_element_type();

        return "<" + Common::get_class_name(self) + ": names[" + names_str + "]" + shape_type_ss.str() + ">";
    });

    output.def_property_readonly("node", &ov::Output<VT>::get_node_shared_ptr);
    output.def_property_readonly("index", &ov::Output<VT>::get_index);
    output.def_property_readonly("any_name", &ov::Output<VT>::get_any_name);
    output.def_property_readonly("names", &ov::Output<VT>::get_names);
    output.def_property_readonly("element_type", &ov::Output<VT>::get_element_type);
    output.def_property_readonly("shape", &ov::Output<VT>::get_shape, py::return_value_policy::copy);
    output.def_property_readonly("partial_shape", &ov::Output<VT>::get_partial_shape, py::return_value_policy::copy);
    output.def_property_readonly("target_inputs", &ov::Output<VT>::get_target_inputs);
    output.def_property_readonly("tensor", &ov::Output<VT>::get_tensor);
    output.def_property_readonly("rt_info",
                                (ov::RTMap&(ov::Output<VT>::*)()) &
                                ov::Output<VT>::get_rt_info,
                                py::return_value_policy::reference_internal);
    output.def_property_readonly("rt_info",
                                (const ov::RTMap&(ov::Output<VT>::*)() const) &
                                ov::Output<VT>::get_rt_info,
                                py::return_value_policy::reference_internal);

    // define functions avaliable only for specific type
    def_type_dependent_functions<VT>(output);
}
