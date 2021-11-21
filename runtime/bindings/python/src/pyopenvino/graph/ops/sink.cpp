// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sink.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/node.hpp"
#include "pyopenvino/graph/ops/sink.hpp"

namespace py = pybind11;

void regclass_graph_op_Sink(py::module m) {
    py::class_<ov::op::Sink, std::shared_ptr<ov::op::Sink>, ov::Node> sink(m, "Sink");

    sink.doc() = "openvino.impl.op.Sink wraps ov::op::Sink";
}
