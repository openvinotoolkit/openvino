// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/axis_set.hpp"  // ov::AxisSet

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/axis_set.hpp"

namespace py = pybind11;

void regclass_graph_AxisSet(py::module m) {
    py::class_<ov::AxisSet, std::shared_ptr<ov::AxisSet>> axis_set(m, "AxisSet");
    axis_set.doc() = "openvino.AxisSet wraps ov::AxisSet";
    axis_set.def(py::init<const std::set<size_t>&>(), py::arg("axes"));
    axis_set.def(py::init<const std::vector<size_t>&>(), py::arg("axes"));
    axis_set.def(py::init<const ov::AxisSet&>(), py::arg("axes"));

    axis_set.def("__len__", [](const ov::AxisSet& v) {
        return v.size();
    });

    axis_set.def(
        "__iter__",
        [](ov::AxisSet& v) {
            return py::make_iterator(v.begin(), v.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    axis_set.def("__repr__", [](const ov::AxisSet& self) -> std::string {
        std::stringstream data_ss;
        std::copy(self.begin(), self.end(), std::ostream_iterator<size_t>(data_ss, ", "));
        std::string data_str = data_ss.str();
        return "<" + Common::get_class_name(self) + " {" + data_str.substr(0, data_str.size() - 2) + "}>";
    });
}
