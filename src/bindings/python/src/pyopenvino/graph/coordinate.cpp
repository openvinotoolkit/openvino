// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate.hpp"  // ov::Coordinate

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/coordinate.hpp"

namespace py = pybind11;

void regclass_graph_Coordinate(py::module m) {
    py::class_<ov::Coordinate, std::shared_ptr<ov::Coordinate>> coordinate(m, "Coordinate");
    coordinate.doc() = "openvino.Coordinate wraps ov::Coordinate";
    coordinate.def(py::init<const ov::Shape&>());
    coordinate.def(py::init<const std::vector<size_t>&>());
    coordinate.def(py::init<const ov::Coordinate&>());
    coordinate.def("__setitem__", [](ov::Coordinate& self, size_t key, size_t value) {
        self[key] = value;
    });

    coordinate.def("__getitem__", [](const ov::Coordinate& self, size_t key) {
        return self[key];
    });

    coordinate.def("__len__", [](const ov::Coordinate& self) {
        return self.size();
    });

    coordinate.def(
        "__iter__",
        [](const ov::Coordinate& self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    coordinate.def("__repr__", [](const ov::Coordinate& self) {
        return Common::get_simple_repr(self);
    });
}
