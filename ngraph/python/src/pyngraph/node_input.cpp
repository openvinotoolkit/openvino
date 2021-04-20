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

    input.def("get_node", &ngraph::Input<ngraph::Node>::get_node);
    input.def("get_index", &ngraph::Input<ngraph::Node>::get_index);
    input.def("get_element_type", &ngraph::Input<ngraph::Node>::get_element_type);
    input.def("get_shape", &ngraph::Input<ngraph::Node>::get_shape);
    input.def("get_partial_shape", &ngraph::Input<ngraph::Node>::get_partial_shape);
    input.def("get_source_output", &ngraph::Input<ngraph::Node>::get_source_output);
}
