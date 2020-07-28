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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "pyngraph/ops/util/index_reduction.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_IndexReduction(py::module m)
{
    py::class_<ngraph::op::util::IndexReduction, std::shared_ptr<ngraph::op::util::IndexReduction>>
        indexReduction(m, "IndexRedection");
    indexReduction.def_property_readonly("reduction_axis",
                                         &ngraph::op::util::IndexReduction::get_reduction_axis);
    indexReduction.def_property_readonly("index_element_type",
                                         &ngraph::op::util::IndexReduction::get_index_element_type);
}
