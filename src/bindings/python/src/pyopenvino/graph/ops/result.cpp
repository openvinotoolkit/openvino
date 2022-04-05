// Copyright (C) 2018-2022 Intel Corporation
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

    result.doc() = "openvino.runtime.op.Result wraps ov::op::v0::Result";

    result.def("get_output_partial_shape", &ov::Node::get_output_partial_shape, py::arg("index"));

    result.def("get_output_element_type", &ov::Node::get_output_element_type, py::arg("index"));

    result.def("get_layout", &ov::op::v0::Result::get_layout);

    result.def("set_layout", &ov::op::v0::Result::set_layout, py::arg("layout"));

    result.def_property("layout", &ov::op::v0::Result::get_layout, &ov::op::v0::Result::set_layout);
}
