// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/strides.hpp"  // ov::Strides

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/strides.hpp"

namespace py = pybind11;

template <typename T>
bool compare_strides(const ov::Strides& a, const T& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](const size_t& elem_a, const py::handle& elem_b) {
               return elem_a == elem_b.cast<size_t>();
           });
}

void regclass_graph_Strides(py::module m) {
    py::class_<ov::Strides, std::shared_ptr<ov::Strides>> strides(m, "Strides");
    strides.doc() = "openvino.Strides wraps ov::Strides";
    strides.def(py::init<const std::vector<size_t>&>(), py::arg("axis_strides"));
    strides.def(py::init<const ov::Strides&>(), py::arg("axis_strides"));

    strides.def("__str__", [](const ov::Strides& self) -> std::string {
        std::stringstream stringstream;
        std::copy(self.begin(), self.end(), std::ostream_iterator<size_t>(stringstream, ", "));
        std::string string = stringstream.str();
        return string.substr(0, string.size() - 2);
    });

    strides.def("__repr__", [](const ov::Strides& self) -> std::string {
        std::string class_name = Common::get_class_name(self);
        std::string shape_str = py::cast(self).attr("__str__")().cast<std::string>();
        return "<" + class_name + ": (" + shape_str + ")>";
    });

    strides.def("__setitem__", [](ov::Strides& self, size_t key, size_t value) {
        self[key] = value;
    });

    strides.def("__getitem__", [](const ov::Strides& self, size_t key) {
        return self[key];
    });

    strides.def("__len__", [](const ov::Strides& self) {
        return self.size();
    });

    strides.def(
        "__eq__",
        [](const ov::Strides& a, const ov::Strides& b) {
            return a == b;
        },
        py::is_operator());

    strides.def(
        "__eq__",
        [](const ov::Strides& a, const py::tuple& b) {
            return compare_strides<py::tuple>(a, b);
        },
        py::is_operator());

    strides.def(
        "__eq__",
        [](const ov::Strides& a, const py::list& b) {
            return compare_strides<py::list>(a, b);
        },
        py::is_operator());

    strides.def(
        "__iter__",
        [](const ov::Strides& self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */
}
