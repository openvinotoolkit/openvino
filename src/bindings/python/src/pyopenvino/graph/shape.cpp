// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"  // ov::Shape

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "openvino/core/dimension.hpp"  // ov::Dimension
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/shape.hpp"

namespace py = pybind11;

void regclass_graph_Shape(py::module m) {
    py::class_<ov::Shape, std::shared_ptr<ov::Shape>> shape(m, "Shape");
    shape.doc() = "openvino.runtime.Shape wraps ov::Shape";
    shape.def(py::init<>());
    shape.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axis_lengths"));
    shape.def(py::init<const std::vector<size_t>&>(), py::arg("axis_lengths"));
    shape.def(py::init<const ov::Shape&>(), py::arg("axis_lengths"));
    shape.def(py::init<const std::string&>(), py::arg("shape"));
    shape.def(
        "__eq__",
        [](const ov::Shape& a, const ov::Shape& b) {
            return a == b;
        },
        py::is_operator());
    shape.def("__len__", [](const ov::Shape& v) {
        return v.size();
    });
    shape.def("__setitem__", [](ov::Shape& self, size_t key, ov::Dimension::value_type d) {
        self[key] = d;
    });
    shape.def("__setitem__", [](ov::Shape& self, size_t key, ov::Dimension d) {
        self[key] = d.get_length();
    });
    shape.def("__getitem__", [](const ov::Shape& v, size_t key) {
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
        return "<" + Common::get_class_name(self) + ": " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });

    shape.def("to_string", &ov::Shape::to_string);
}
