// Copyright (C) 2018-2023 Intel Corporation
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

void regclass_graph_Strides(py::module m) {
    py::class_<ov::Strides, std::shared_ptr<ov::Strides>> strides(m, "Strides");
    strides.doc() = "openvino.runtime.Strides wraps ov::Strides";
    strides.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axis_strides"));
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
        "__iter__",
        [](const ov::Strides& self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */
}
