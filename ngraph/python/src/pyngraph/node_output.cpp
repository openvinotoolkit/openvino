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

    output.def("get_node", &ngraph::Output<ngraph::Node>::get_node);
    output.def("get_index", &ngraph::Output<ngraph::Node>::get_index);
    output.def("get_element_type", &ngraph::Output<ngraph::Node>::get_element_type);
    output.def("get_shape", &ngraph::Output<ngraph::Node>::get_shape);
    output.def("get_partial_shape", &ngraph::Output<ngraph::Node>::get_partial_shape);
    output.def("get_target_inputs", &ngraph::Output<ngraph::Node>::get_target_inputs);
}
