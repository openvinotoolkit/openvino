// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/assign.hpp"

#include <pybind11/pybind11.h>

#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_Assign(py::module m) {
    py::class_<ov::op::v6::Assign, std::shared_ptr<ov::op::v6::Assign>, ov::Node> assign(m, "assign");

    assign.doc() = "openvino.runtime.op.assign wraps ov::op::v6::Assign";

    assign.def(py::init<>());

    assign.def(py::init([](py::object& new_value, const std::string& variable_id, const std::string& name) {
                   auto node = new_value.cast<std::shared_ptr<ov::Node>>();

                   auto variable = std::make_shared<ov::op::util::Variable>(
                       ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_id});
                   return std::make_shared<ov::op::v6::Assign>(node, variable);
               }),
               py::arg("new_value"),
               py::arg("variable_id"),
               py::arg("name") = "");

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
