// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate.hpp"  // ngraph::Coordinate

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyngraph/coordinate.hpp"

namespace py = pybind11;

void regclass_pyngraph_Coordinate(py::module m) {
    py::class_<ngraph::Coordinate, std::shared_ptr<ngraph::Coordinate>> coordinate(m, "Coordinate", py::module_local());
    coordinate.doc() = "ngraph.impl.Coordinate wraps ngraph::Coordinate";
    coordinate.def(py::init<const std::initializer_list<size_t>&>());
    coordinate.def(py::init<const ngraph::Shape&>());
    coordinate.def(py::init<const std::vector<size_t>&>());
    coordinate.def(py::init<const ngraph::Coordinate&>());
}
