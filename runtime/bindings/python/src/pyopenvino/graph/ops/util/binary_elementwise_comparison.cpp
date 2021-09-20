// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/binary_elementwise_comparison.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/graph/ops/util/binary_elementwise_comparison.hpp"

namespace py = pybind11;

void regclass_graph_op_util_BinaryElementwiseComparison(py::module m) {
    py::class_<ngraph::op::util::BinaryElementwiseComparison,
               std::shared_ptr<ngraph::op::util::BinaryElementwiseComparison>>
        binaryElementwiseComparison(m, "BinaryElementwiseComparison");
}
