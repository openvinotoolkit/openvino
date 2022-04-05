// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/result.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "ngraph/node.hpp"
#include "pyngraph/ops/result.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Result(py::module m) {
    py::class_<ngraph::op::Result, std::shared_ptr<ngraph::op::Result>, ngraph::Node> result(m,
                                                                                             "Result",
                                                                                             py::module_local());
    result.doc() = "ngraph.impl.op.Result wraps ngraph::op::Result";
}
