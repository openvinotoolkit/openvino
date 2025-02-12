// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/util/variable.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/op/util/variable.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_util_Variable(py::module m) {
    py::class_<ov::op::util::VariableInfo> variable_info(m, "VariableInfo");
    variable_info.doc() = "openvino.op.util.VariableInfo wraps ov::op::util::VariableInfo";
    variable_info.def(py::init<>());
    variable_info.def_readwrite("data_shape", &ov::op::util::VariableInfo::data_shape);
    variable_info.def_readwrite("data_type", &ov::op::util::VariableInfo::data_type);
    variable_info.def_readwrite("variable_id", &ov::op::util::VariableInfo::variable_id);
    variable_info.def("__repr__", [](const ov::op::util::VariableInfo& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::op::util::Variable, std::shared_ptr<ov::op::util::Variable>> variable(m, "Variable");
    variable.doc() = "openvino.op.util.Variable wraps ov::op::util::Variable";
    variable.def(py::init([](const ov::op::util::VariableInfo& info) {
                     return ov::op::util::Variable{info};
                 }),
                 py::arg("info"));
    variable.def_property_readonly("info", &ov::op::util::Variable::get_info);
    variable.def("get_info", &ov::op::util::Variable::get_info);
    variable.def("update", &ov::op::util::Variable::update, py::arg("variable_info"));
    variable.def("__repr__", [](const ov::op::util::Variable& self) {
        return Common::get_simple_repr(self);
    });
}
