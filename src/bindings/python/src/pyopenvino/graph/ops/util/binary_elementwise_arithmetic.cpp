// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/binary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regclass_graph_op_util_BinaryElementwiseArithmetic(py::module m) {
    py::class_<ov::op::util::BinaryElementwiseArithmetic, std::shared_ptr<ov::op::util::BinaryElementwiseArithmetic>>
        binaryElementwiseArithmetic(m, "BinaryElementwiseArithmetic");
    binaryElementwiseArithmetic.def("__repr__", [](const ov::op::util::BinaryElementwiseArithmetic& self) {
        return Common::get_simple_repr(self);
    });
}
