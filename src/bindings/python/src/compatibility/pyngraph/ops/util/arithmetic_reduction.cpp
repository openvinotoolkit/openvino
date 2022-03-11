// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/arithmetic_reduction.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/op.hpp"
#include "pyngraph/ops/util/arithmetic_reduction.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_ArithmeticReduction(py::module m) {
    py::class_<ngraph::op::util::ArithmeticReduction, std::shared_ptr<ngraph::op::util::ArithmeticReduction>>
        arithmeticReduction(m, "ArithmeticReduction", py::module_local());
    // arithmeticReduction.def(py::init<const std::string&,
    //                                  const std::shared_ptr<ngraph::Node>&,
    //                                  const ngraph::AxisSet& >());
    arithmeticReduction.def("get_reduction_axes", &ngraph::op::util::ArithmeticReduction::get_reduction_axes);
    arithmeticReduction.def("set_reduction_axes", &ngraph::op::util::ArithmeticReduction::set_reduction_axes);

    arithmeticReduction.def_property("reduction_axes",
                                     &ngraph::op::util::ArithmeticReduction::get_reduction_axes,
                                     &ngraph::op::util::ArithmeticReduction::set_reduction_axes);
}
