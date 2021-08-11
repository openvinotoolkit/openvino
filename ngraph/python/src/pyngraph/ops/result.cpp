// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "ngraph/node.hpp"
#include "ngraph/op/result.hpp"
#include "pyngraph/ops/result.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Result(py::module m)
{
    py::class_<ov::op::Result, std::shared_ptr<ov::op::Result>, ov::Node> result(m, "Result");
    result.doc() = "ngraph.impl.op.Result wraps ov::op::Result";
}
