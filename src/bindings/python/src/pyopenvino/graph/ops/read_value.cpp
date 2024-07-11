// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/read_value.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_ReadValue(py::module m) {
    py::class_<ov::op::v6::ReadValue, std::shared_ptr<ov::op::v6::ReadValue>, ov::Node> read_value(m, "read_value");

    read_value.doc() = "openvino.runtime.op.read_value wraps ov::op::v6::ReadValue";

    read_value.def(py::init<>());

    read_value.def(py::init([](const py::object& new_value, const std::shared_ptr<ov::op::util::Variable>& variable) {
                       if (py::isinstance<ov::Node>(new_value)) {
                           auto node = new_value.cast<std::shared_ptr<ov::Node>>();
                           return std::make_shared<ov::op::v6::ReadValue>(node, variable);
                       } else if (py::isinstance<const ov::Output<ov::Node>>(new_value) ||
                                  py::isinstance<ov::Output<ov::Node>>(new_value)) {
                           auto output = new_value.cast<const ov::Output<ov::Node>>();
                           return std::make_shared<ov::op::v6::ReadValue>(output, variable);
                       } else {
                           throw py::type_error("Wrong type");
                       }
                   }),
                   py::arg("new_value"),
                   py::arg("variable"));

    read_value.def(py::init([](const std::shared_ptr<ov::op::util::Variable>& variable) {
                       return std::make_shared<ov::op::v6::ReadValue>(variable);
                   }),
                   py::arg("variable"));

    read_value.def(
        "get_variable_id",
        [](ov::op::v6::ReadValue& self) {
            return self.get_variable_id();
        },
        R"(
            Gets variable id.

            :return: variable id.
            :rtype: str
        )");

    read_value.def("__repr__", [](ov::op::v6::ReadValue& self) {
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
