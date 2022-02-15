// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

#include <pybind11/pybind11.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/any.hpp"

namespace py = pybind11;

void regclass_graph_Any(py::module m) {
    py::class_<PyAny, std::shared_ptr<PyAny>> ov_any(m, "OVAny", py::module_local());
    ov_any.doc() = "openvino.runtime.OVAny provides object wrapper for OpenVINO"
                   "ov::Any class. It allows to pass different types of objects"
                   "into C++ based core of the project.";

    ov_any.def(py::init<py::object>());

    ov_any.def("__repr__", [](const PyAny& self) {
        std::stringstream ret;
        self.print(ret);
        return ret.str();
    });
    ov_any.def("__eq__", [](const PyAny& a, const PyAny& b) -> bool {
        return a == b;
    });
    ov_any.def("__eq__", [](const PyAny& a, const ov::Any& b) -> bool {
        return a == b;
    });
    ov_any.def("__eq__", [](const PyAny& a, py::object b) -> bool {
        return a == PyAny(b);
    });
    ov_any.def(
        "get",
        [](const PyAny& self) -> py::object {
            return self.as<py::object>();
        },
        R"(
            :return: Value of this OVAny.
            :rtype: Any
        )");
    ov_any.def(
        "set",
        [](PyAny& self, py::object value) {
            self = PyAny(value);
        },
        R"(
            :param: Value to be set in OVAny.
            :type: Any
    )");
    ov_any.def_property_readonly(
        "value",
        [](const PyAny& self) {
            return self.as<py::object>();
        },
        R"(
            :return: Value of this OVAny.
            :rtype: Any
    )");
}
