// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/strides.hpp"  // ov::Strides

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/graph/strides.hpp"

namespace py = pybind11;

void regclass_graph_Strides(py::module m) {
    py::class_<ov::Strides, std::shared_ptr<ov::Strides>> strides(m, "Strides");
    strides.doc() = "openvino.impl.Strides wraps ov::Strides";
    strides.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axis_strides"));
    strides.def(py::init<const std::vector<size_t>&>(), py::arg("axis_strides"));
    strides.def(py::init<const ov::Strides&>(), py::arg("axis_strides"));

    strides.def("__str__", [](const ov::Strides& self) -> std::string {
        std::stringstream stringstream;
        std::copy(self.begin(), self.end(), std::ostream_iterator<int>(stringstream, ", "));
        std::string string = stringstream.str();
        return string.substr(0, string.size() - 2);
    });

    strides.def("__repr__", [](const ov::Strides& self) -> std::string {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::string shape_str = py::cast(self).attr("__str__")().cast<std::string>();
        return "<" + class_name + ": (" + shape_str + ")>";
    });
}
