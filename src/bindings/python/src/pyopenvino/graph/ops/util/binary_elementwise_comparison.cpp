// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/binary_elementwise_comparison.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/binary_elementwise_comparison.hpp"

namespace py = pybind11;

void regclass_graph_op_util_BinaryElementwiseComparison(py::module m) {
    py::class_<ov::op::util::BinaryElementwiseComparison, std::shared_ptr<ov::op::util::BinaryElementwiseComparison>>
        binaryElementwiseComparison(m, "BinaryElementwiseComparison");
    binaryElementwiseComparison.def("__repr__", [](const ov::op::util::BinaryElementwiseComparison& self) {
        return Common::get_simple_repr(self);
    });
}
