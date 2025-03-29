// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/binary_elementwise_logical.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/binary_elementwise_logical.hpp"

namespace py = pybind11;

void regclass_graph_op_util_BinaryElementwiseLogical(py::module m) {
    py::class_<ov::op::util::BinaryElementwiseLogical, std::shared_ptr<ov::op::util::BinaryElementwiseLogical>>
        binaryElementwiseLogical(m, "BinaryElementwiseLogical");
    binaryElementwiseLogical.def("__repr__", [](const ov::op::util::BinaryElementwiseLogical& self) {
        return Common::get_simple_repr(self);
    });
}
