// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate.hpp"  // ov::Coordinate

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/shape.hpp"
#include "pyngraph/coordinate.hpp"

namespace py = pybind11;

void regclass_pyngraph_Coordinate(py::module m) {
    py::class_<ov::Coordinate, std::shared_ptr<ov::Coordinate>> coordinate(m, "Coordinate", py::module_local());
    coordinate.doc() = "ngraph.impl.Coordinate wraps ov::Coordinate";
    coordinate.def(py::init<const std::initializer_list<size_t>&>());
    coordinate.def(py::init<const ov::Shape&>());
    coordinate.def(py::init<const std::vector<size_t>&>());
    coordinate.def(py::init<const ov::Coordinate&>());
}
