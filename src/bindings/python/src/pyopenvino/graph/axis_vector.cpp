// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/axis_vector.hpp"  // ov::AxisVector

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/axis_vector.hpp"

namespace py = pybind11;

void regclass_graph_AxisVector(py::module m) {
    py::class_<ov::AxisVector, std::shared_ptr<ov::AxisVector>> axis_vector(m, "AxisVector");
    axis_vector.doc() = "openvino.AxisVector wraps ov::AxisVector";
    axis_vector.def(py::init<const std::vector<size_t>&>(), py::arg("axes"));
    axis_vector.def(py::init<const ov::AxisVector&>(), py::arg("axes"));
    axis_vector.def("__setitem__", [](ov::AxisVector& self, size_t key, size_t value) {
        self[key] = value;
    });

    axis_vector.def("__getitem__", [](const ov::AxisVector& self, size_t key) {
        return self[key];
    });

    axis_vector.def("__len__", [](const ov::AxisVector& self) {
        return self.size();
    });

    axis_vector.def(
        "__iter__",
        [](const ov::AxisVector& self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    axis_vector.def("__repr__", [](const ov::AxisVector& self) {
        std::stringstream data_ss;
        std::copy(self.begin(), self.end(), std::ostream_iterator<size_t>(data_ss, ", "));
        std::string data_str = data_ss.str();
        return "<" + Common::get_class_name(self) + " {" + data_str.substr(0, data_str.size() - 2) + "}>";
    });
}
