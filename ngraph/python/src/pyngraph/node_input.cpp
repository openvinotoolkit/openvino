// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/node_input.hpp"
#include "pyngraph/node_input.hpp"

namespace py = pybind11;

void regclass_pyngraph_Input(py::module m)
{
    py::class_<ngraph::Input<ngraph::Node>, std::shared_ptr<ngraph::Input<ngraph::Node>>> input(
        m, "Input", py::dynamic_attr());
    input.doc() = "ngraph.impl.Input wraps ngraph::Input<Node>";

    input.def("get_node",
              &ngraph::Input<ngraph::Node>::get_node,
              R"(
                Get node referenced by this input handle.

                Returns
                ----------
                get_node : Node
                    Node object referenced by this input handle.
              )");
    input.def("get_index",
              &ngraph::Input<ngraph::Node>::get_index,
              R"(
                The index of the input referred to by this input handle.

                Returns
                ----------
                get_index : int
                    Index value as integer.
              )");
    input.def("get_element_type",
              &ngraph::Input<ngraph::Node>::get_element_type,
              R"(
                The element type of the input referred to by this input handle.

                Returns
                ----------
                get_element_type : Type
                    Type of the input.
              )");
    input.def("get_shape",
              &ngraph::Input<ngraph::Node>::get_shape,
              R"(
                The shape of the input referred to by this input handle.

                Returns
                ----------
                get_shape : Shape
                    Shape of the input.
              )");
    input.def("get_partial_shape",
              &ngraph::Input<ngraph::Node>::get_partial_shape,
              R"(
                The partial shape of the input referred to by this input handle.

                Returns
                ----------
                get_partial_shape : PartialShape
                    PartialShape of the input.
              )");
    input.def("get_source_output",
              &ngraph::Input<ngraph::Node>::get_source_output,
              R"(
                A handle to the output that is connected to this input.

                Returns
                ----------
                get_source_output : Output
                    Output that is connected to the input.
              )");
}
