//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/get_output_element.hpp"
#include "pyngraph/ops/get_output_element.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_GetOutputElement(py::module m)
{
    py::class_<ngraph::op::GetOutputElement,
               std::shared_ptr<ngraph::op::GetOutputElement>,
               ngraph::op::Op>
        get_output_element(m, "GetOutputElement");
    get_output_element.doc() = "ngraph.impl.op.GetOutputElement wraps ngraph::op::GetOutputElement";
    get_output_element.def(py::init<const std::shared_ptr<ngraph::Node>&, size_t>());
}
