// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate.hpp"  // ov::Coordinate

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/graph/coordinate.hpp"

namespace py = pybind11;

void regclass_graph_Coordinate(py::module m) {
    py::class_<ov::Coordinate, std::shared_ptr<ov::Coordinate>> coordinate(m, "Coordinate");
    coordinate.doc() = "ngraph.impl.Coordinate wraps ov::Coordinate";
    coordinate.def(py::init<const std::initializer_list<size_t>&>());
    coordinate.def(py::init<const ov::Shape&>());
    coordinate.def(py::init<const std::vector<size_t>&>());
    coordinate.def(py::init<const ov::Coordinate&>());
}
