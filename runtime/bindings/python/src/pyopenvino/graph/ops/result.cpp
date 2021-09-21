// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/core/node.hpp"
#include "pyopenvino/graph/ops/result.hpp"

namespace py = pybind11;

void regclass_graph_op_Result(py::module m) {
    py::class_<ov::op::v0::Result, std::shared_ptr<ov::op::v0::Result>, ov::Node> result(m, "Result");

    result.doc() = "openvino.impl.op.Result wraps ov::op::v0::Result";
}
