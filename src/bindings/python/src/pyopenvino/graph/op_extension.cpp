// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/op_extension.hpp"

#include <pybind11/pybind11.h>

#include <pyopenvino/graph/op_extension.hpp>
#include "pyopenvino/graph/op.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_OpExtension(py::module m) {
    py::class_<ov::OpExtension<PyOp>, std::shared_ptr<ov::OpExtension<PyOp>>, ov::Extension> op_extension(m, "OpExtension");
    op_extension.doc() = "openvino.OpExtension provides the base interface for OpenVINO extensions.";

    op_extension.def("__repr__", [](const ov::OpExtension<PyOp>& self) {
        return Common::get_simple_repr(self);
    });

    op_extension.def(py::init<>());
    op_extension.def(py::init<>());
}
