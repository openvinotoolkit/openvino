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
#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "pyngraph/ops/util/arithmetic_reduction.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_ArithmeticReduction(py::module m)
{
    py::class_<ngraph::op::util::ArithmeticReduction,
               std::shared_ptr<ngraph::op::util::ArithmeticReduction>>
        arithmeticReduction(m, "ArithmeticReduction");
    // arithmeticReduction.def(py::init<const std::string&,
    //                                  const std::shared_ptr<ngraph::Node>&,
    //                                  const ngraph::AxisSet& >());
    arithmeticReduction.def("get_reduction_axes",
                            &ngraph::op::util::ArithmeticReduction::get_reduction_axes);
    arithmeticReduction.def("set_reduction_axes",
                            &ngraph::op::util::ArithmeticReduction::set_reduction_axes);

    arithmeticReduction.def_property("reduction_axes",
                                     &ngraph::op::util::ArithmeticReduction::get_reduction_axes,
                                     &ngraph::op::util::ArithmeticReduction::set_reduction_axes);
}
