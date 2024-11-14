// Copyright (C) 2018-2024 Intel Corporation
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

template <typename T>
bool compare_shape(const ov::Shape& a, const T& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](const ov::Dimension& elem_a, const py::handle& elem_b) {
               return elem_a == elem_b.cast<int64_t>();
           });
}

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
    shape.def(
        "__eq__",
        [](const ov::Shape& a, const py::tuple& b) {
            return compare_shape<py::tuple>(a, b);
        },
        py::is_operator());
    shape.def(
        "__eq__",
        [](const ov::Shape& a, const py::list& b) {
            return compare_shape<py::list>(a, b);
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
    shape.def("__getitem__", [](const ov::Shape& v, int64_t key) {
        if (key < 0) {
            key += v.size();
        }
        return v[key];
    });

    shape.def("__getitem__", [](const ov::Shape& v, py::slice& slice) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
        }
        ov::Shape result(slicelength);
        Common::shape_helpers::get_slice(result, v, start, step, slicelength);
        return result;
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
