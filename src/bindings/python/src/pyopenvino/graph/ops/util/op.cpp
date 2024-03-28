// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "op.hpp"

namespace py = pybind11;

void regclass_graph_op_util_Op(py::module m) {
    py::class_<ov::op::Op, ov::Node, PyOp, std::shared_ptr<ov::op::Op>>(m, "_PyOp")
        .def(py::init<>());
}
