// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyngraph/ops/util/unary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_UnaryElementwiseArithmetic(py::module m) {
    py::class_<ngraph::op::util::UnaryElementwiseArithmetic,
               std::shared_ptr<ngraph::op::util::UnaryElementwiseArithmetic>>
        unaryElementwiseArithmetic(m, "UnaryElementwiseArithmetic", py::module_local());
}
