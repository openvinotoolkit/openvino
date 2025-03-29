// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/unary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regclass_graph_op_util_UnaryElementwiseArithmetic(py::module m) {
    py::class_<ov::op::util::UnaryElementwiseArithmetic, std::shared_ptr<ov::op::util::UnaryElementwiseArithmetic>>
        unaryElementwiseArithmetic(m, "UnaryElementwiseArithmetic");
    unaryElementwiseArithmetic.def("__repr__", [](const ov::op::util::UnaryElementwiseArithmetic& self) {
        return Common::get_simple_repr(self);
    });
}
