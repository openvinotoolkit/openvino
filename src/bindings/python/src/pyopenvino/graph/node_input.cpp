// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_input.hpp"

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "pyopenvino/graph/node_input.hpp"

namespace py = pybind11;

void regclass_graph_Input(py::module m) {
    py::class_<ov::Input<ov::Node>, std::shared_ptr<ov::Input<ov::Node>>> input(m, "Input", py::dynamic_attr());
    input.doc() = "openvino.runtime.Input wraps ov::Input<Node>";

    input.def("get_node",
              &ov::Input<ov::Node>::get_node,
              R"(
                Get node referenced by this input handle.

                Returns
                ----------
                get_node : Node
                    Node object referenced by this input handle.
              )");
    input.def("get_index",
              &ov::Input<ov::Node>::get_index,
              R"(
                The index of the input referred to by this input handle.

                Returns
                ----------
                get_index : int
                    Index value as integer.
              )");
    input.def("get_element_type",
              &ov::Input<ov::Node>::get_element_type,
              R"(
                The element type of the input referred to by this input handle.

                Returns
                ----------
                get_element_type : Type
                    Type of the input.
              )");
    input.def("get_shape",
              &ov::Input<ov::Node>::get_shape,
              R"(
                The shape of the input referred to by this input handle.

                Returns
                ----------
                get_shape : Shape
                    Shape of the input.
              )");
    input.def("get_partial_shape",
              &ov::Input<ov::Node>::get_partial_shape,
              R"(
                The partial shape of the input referred to by this input handle.

                Returns
                ----------
                get_partial_shape : PartialShape
                    PartialShape of the input.
              )");
    input.def("get_source_output",
              &ov::Input<ov::Node>::get_source_output,
              R"(
                A handle to the output that is connected to this input.

                Returns
                ----------
                get_source_output : Output
                    Output that is connected to the input.
              )");
}
