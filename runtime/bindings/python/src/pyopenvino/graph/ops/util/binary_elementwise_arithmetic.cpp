// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/graph/ops/util/binary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regclass_graph_op_util_BinaryElementwiseArithmetic(py::module m) {
    py::class_<ngraph::op::util::BinaryElementwiseArithmetic,
               std::shared_ptr<ngraph::op::util::BinaryElementwiseArithmetic>>
        binaryElementwiseArithmetic(m, "BinaryElementwiseArithmetic", py::module_local());
}
