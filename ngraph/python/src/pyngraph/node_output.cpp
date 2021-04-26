// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/node_output.hpp"
#include "pyngraph/node_output.hpp"

namespace py = pybind11;

void regclass_pyngraph_Output(py::module m)
{
    py::class_<ngraph::Output<ngraph::Node>, std::shared_ptr<ngraph::Output<ngraph::Node>>> output(
        m, "Output", py::dynamic_attr());
    output.doc() = "ngraph.impl.Output wraps ngraph::Output<Node>";

    output.def("get_node",
               &ngraph::Output<ngraph::Node>::get_node,
               R"mydelimiter(
                Get node referenced by this output handle.

                Returns
                ----------
                get_node : Node
                    Node object referenced by this output handle.
               )mydelimiter");
    output.def("get_index",
               &ngraph::Output<ngraph::Node>::get_index,
               R"mydelimiter(
                The index of the output referred to by this output handle.

                Returns
                ----------
                get_index : int
                    Index value as integer.
               )mydelimiter");
    output.def("get_element_type",
               &ngraph::Output<ngraph::Node>::get_element_type,
               R"mydelimiter(
                The element type of the output referred to by this output handle.

                Returns
                ----------
                get_element_type : Type
                    Type of the output.
               )mydelimiter");
    output.def("get_shape",
               &ngraph::Output<ngraph::Node>::get_shape,
               R"mydelimiter(
                The shape of the output referred to by this output handle.

                Returns
                ----------
                get_shape : Shape
                    Shape of the output.
               )mydelimiter");
    output.def("get_partial_shape",
               &ngraph::Output<ngraph::Node>::get_partial_shape,
               R"mydelimiter(
                The partial shape of the output referred to by this output handle.

                Returns
                ----------
                get_partial_shape : PartialShape
                    PartialShape of the output.
               )mydelimiter");
    output.def("get_target_inputs",
               &ngraph::Output<ngraph::Node>::get_target_inputs,
               R"mydelimiter(
                A set containing handles for all inputs targeted by the output
                referenced by this output handle.
                Returns
                ----------
                get_target_inputs : Set[Input]
                    Set of Inputs.
               )mydelimiter");
}
