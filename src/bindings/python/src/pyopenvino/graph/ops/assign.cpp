// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/assign.hpp"

#include <pybind11/pybind11.h>

#include "openvino/op/sink.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_Assign(py::module m) {
    py::class_<ov::op::v6::Assign, std::shared_ptr<ov::op::v6::Assign>, ov::Node> assign(m, "Assign");

    assign.doc() = "openvino.runtime.op.Assign wraps ov::op::v6::Assign";

    assign.def(py::init<>());

    assign.def("__repr__", [](ov::op::v6::Assign& self) {
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i) {
            if (i > 0) {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + Common::get_class_name(self) + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });
}