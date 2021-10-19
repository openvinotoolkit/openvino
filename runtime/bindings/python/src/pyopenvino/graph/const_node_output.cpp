// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/const_node_output.hpp"

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/node_output.hpp"

namespace py = pybind11;

void regclass_graph_ConstOutput(py::module m) {
    py::class_<ov::Output<const ov::Node>, std::shared_ptr<ov::Output<const ov::Node>>> output(m,
                                                                                               "ConstOutput",
                                                                                               py::dynamic_attr());
    output.doc() = "openvino.impl.ConstOutput wraps ov::Output<const ov::Node>";

    output.def_property_readonly("get_node",
                                 &ov::Output<const ov::Node>::get_node,
                                 R"(
                Get node referenced by this output handle.

                Returns
                ----------
                get_node : const ov::Node
                    Node object referenced by this output handle.
               )");
    output.def_property_readonly("get_index",
                                 &ov::Output<const ov::Node>::get_index,
                                 R"(
                The index of the output referred to by this output handle.

                Returns
                ----------
                get_index : int
                    Index value as integer.
               )");
    output.def_property_readonly("get_element_type",
                                 &ov::Output<const ov::Node>::get_element_type,
                                 R"(
                The element type of the output referred to by this output handle.

                Returns
                ----------
                get_element_type : const Type
                    Type of the output.
               )");
    output.def_property_readonly("get_shape",
                                 &ov::Output<const ov::Node>::get_shape,
                                 R"(
                The shape of the output referred to by this output handle.

                Returns
                ----------
                get_shape : const Shape
                    Shape of the output.
               )");
    output.def_property_readonly("get_partial_shape",
                                 &ov::Output<const ov::Node>::get_partial_shape,
                                 R"(
                The partial shape of the output referred to by this output handle.

                Returns
                ----------
                get_partial_shape : const PartialShape
                    PartialShape of the output.
               )");
    output.def_property_readonly("get_target_inputs",
                                 &ov::Output<const ov::Node>::get_target_inputs,
                                 R"(
                A set containing handles for all inputs targeted by the output
                referenced by this output handle.
                Returns
                ----------
                get_target_inputs : Set[Input]
                    Set of Inputs.
               )");
}
