//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
