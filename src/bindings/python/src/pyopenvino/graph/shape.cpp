// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"  // ov::Shape

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/graph/shape.hpp"

namespace py = pybind11;

void regclass_graph_Shape(py::module m) {
    py::class_<ov::Shape, std::shared_ptr<ov::Shape>> shape(m, "Shape");
    shape.doc() = "openvino.impl.Shape wraps ov::Shape";
    shape.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axis_lengths"));
    shape.def(py::init<const std::vector<size_t>&>(), py::arg("axis_lengths"));
    shape.def(py::init<const ov::Shape&>(), py::arg("axis_lengths"));
    shape.def("__len__", [](const ov::Shape& v) {
        return v.size();
    });
    shape.def("__getitem__", [](const ov::Shape& v, int key) {
        return v[key];
    });

    shape.def(
        "__iter__",
        [](ov::Shape& v) {
            return py::make_iterator(v.begin(), v.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    shape.def("__str__", [](const ov::Shape& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    shape.def("__repr__", [](const ov::Shape& self) -> std::string {
        return "<Shape: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
