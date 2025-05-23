// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate_diff.hpp"  // ov::CoordinateDiff

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/coordinate_diff.hpp"

namespace py = pybind11;

void regclass_graph_CoordinateDiff(py::module m) {
    py::class_<ov::CoordinateDiff, std::shared_ptr<ov::CoordinateDiff>> coordinate_diff(m, "CoordinateDiff");
    coordinate_diff.doc() = "openvino.CoordinateDiff wraps ov::CoordinateDiff";
    coordinate_diff.def(py::init<const std::vector<ptrdiff_t>&>());
    coordinate_diff.def(py::init<const ov::CoordinateDiff&>());

    coordinate_diff.def("__str__", [](const ov::CoordinateDiff& self) -> std::string {
        std::stringstream stringstream;
        std::copy(self.begin(), self.end(), std::ostream_iterator<size_t>(stringstream, ", "));
        std::string string = stringstream.str();
        return string.substr(0, string.size() - 2);
    });

    coordinate_diff.def("__repr__", [](const ov::CoordinateDiff& self) -> std::string {
        std::string class_name = Common::get_class_name(self);
        std::string shape_str = py::cast(self).attr("__str__")().cast<std::string>();
        return "<" + class_name + ": (" + shape_str + ")>";
    });

    coordinate_diff.def("__setitem__", [](ov::CoordinateDiff& self, size_t key, std::ptrdiff_t& value) {
        self[key] = value;
    });

    coordinate_diff.def("__getitem__", [](const ov::CoordinateDiff& self, size_t key) {
        return self[key];
    });

    coordinate_diff.def("__len__", [](const ov::CoordinateDiff& self) {
        return self.size();
    });

    coordinate_diff.def(
        "__iter__",
        [](const ov::CoordinateDiff& self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */
}
