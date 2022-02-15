// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "openvino/core/node_output.hpp"

namespace py = pybind11;

using PyRTMap = ov::Node::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

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

    output.def("__hash__", [](ov::Output<VT>& port) {
        return std::hash<VT*>()(port.get_node()) + port.get_index();
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
                A set containing handles for all inputs targeted by the output
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
                :rtype: openvino.pyopenvino.DescriptorTensor 
               )");
    output.def("get_rt_info",
             (ov::RTMap & (ov::Output<VT>::*)()) &  ov::Output<VT>::get_rt_info,
             py::return_value_policy::reference_internal,
             R"(
                Returns RTMap which is a dictionary of user defined runtime info.

                :return: A dictionary of user defined data.
                :rtype: openvino.runtime.RTMap
             )");


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
}
