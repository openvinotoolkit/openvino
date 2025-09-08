// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/arithmetic_reduction.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/op/op.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/arithmetic_reduction.hpp"

namespace py = pybind11;

void regclass_graph_op_util_ArithmeticReduction(py::module m) {
    py::class_<ov::op::util::ArithmeticReduction, std::shared_ptr<ov::op::util::ArithmeticReduction>>
        arithmeticReduction(m, "ArithmeticReduction");
    // arithmeticReduction.def(py::init<const std::string&,
    //                                  const std::shared_ptr<ov::Node>&,
    //                                  const ov::AxisSet& >());
    arithmeticReduction.def("get_reduction_axes", &ov::op::util::ArithmeticReduction::get_reduction_axes);
    arithmeticReduction.def("set_reduction_axes", &ov::op::util::ArithmeticReduction::set_reduction_axes);

    arithmeticReduction.def_property("reduction_axes",
                                     &ov::op::util::ArithmeticReduction::get_reduction_axes,
                                     &ov::op::util::ArithmeticReduction::set_reduction_axes);
    arithmeticReduction.def("__repr__", [](const ov::op::util::ArithmeticReduction& self) {
        return Common::get_simple_repr(self);
    });
}
