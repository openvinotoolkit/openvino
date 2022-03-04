// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/binary_elementwise_logical.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyngraph/ops/util/binary_elementwise_logical.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_BinaryElementwiseLogical(py::module m) {
    py::class_<ngraph::op::util::BinaryElementwiseLogical, std::shared_ptr<ngraph::op::util::BinaryElementwiseLogical>>
        binaryElementwiseLogical(m, "BinaryElementwiseLogical", py::module_local());
}
