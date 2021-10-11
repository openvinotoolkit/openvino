// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/axis_vector.hpp"  // ov::AxisVector

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/graph/axis_vector.hpp"

namespace py = pybind11;

void regclass_graph_AxisVector(py::module m) {
    py::class_<ov::AxisVector, std::shared_ptr<ov::AxisVector>> axis_vector(m, "AxisVector");
    axis_vector.doc() = "ngraph.impl.AxisVector wraps ov::AxisVector";
    axis_vector.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axes"));
    axis_vector.def(py::init<const std::vector<size_t>&>(), py::arg("axes"));
    axis_vector.def(py::init<const ov::AxisVector&>(), py::arg("axes"));
}
