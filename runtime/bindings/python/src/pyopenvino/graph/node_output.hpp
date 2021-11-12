// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/node_output.hpp"

namespace py = pybind11;

template <typename VT>
void regclass_graph_Output(py::module m, std::string typestring)
{
    auto pyclass_name = py::detail::c_str((typestring + std::string("Output")));
    auto docs = py::detail::c_str((std::string("openvino.impl.") + typestring + std::string("Output wraps ov::Output<") + typestring + std::string(" ov::Node >")));
    py::class_<ov::Output<VT>, std::shared_ptr<ov::Output<VT>>> output(m,
                                                                       pyclass_name,
                                                                       py::dynamic_attr());
    output.doc() = docs;

    output.def("get_node",
               &ov::Output<VT>::get_node,
               R"(
                Get node referenced by this output handle.

                Returns
                ----------
                get_node : Node or const Node
                    Node object referenced by this output handle.
               )");
    output.def("get_index",
               &ov::Output<VT>::get_index,
               R"(
                The index of the output referred to by this output handle.

                Returns
                ----------
                get_index : int
                    Index value as integer.
               )");
    output.def("get_element_type",
               &ov::Output<VT>::get_element_type,
               R"(
                The element type of the output referred to by this output handle.

                Returns
                ----------
                get_element_type : Type
                    Type of the output.
               )");
    output.def("get_shape",
               &ov::Output<VT>::get_shape,
               R"(
                The shape of the output referred to by this output handle.

                Returns
                ----------
                get_shape : Shape
                    Shape of the output.
               )");
    output.def("get_partial_shape",
               &ov::Output<VT>::get_partial_shape,
               R"(
                The partial shape of the output referred to by this output handle.

                Returns
                ----------
                get_partial_shape : PartialShape
                    PartialShape of the output.
               )");
    output.def("get_target_inputs",
               &ov::Output<VT>::get_target_inputs,
               R"(
                A set containing handles for all inputs targeted by the output
                referenced by this output handle.
                Returns
                ----------
                get_target_inputs : Set[Input]
                    Set of Inputs.
               )");
}
