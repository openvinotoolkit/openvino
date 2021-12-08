// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/variant.hpp"

#include <pybind11/pybind11.h>

#include "openvino/core/any.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_Variant(py::module m) {
    py::class_<PyAny, std::shared_ptr<PyAny>> variant(m, "Variant", py::module_local());
    variant.doc() = "openvino.impl.Variant wraps ov::Any";
    variant.def(py::init<py::object>());

    variant.def("__repr__", [](const PyAny& self) {
        std::stringstream ret;
        self.print(ret);
        return ret.str();
    });
    variant.def("__eq__", [](const PyAny& a, const PyAny& b) -> bool {
        return a == b;
    });
    variant.def("__eq__", [](const PyAny& a, const ov::Any& b) -> bool {
        return a == b;
    });
    variant.def("__eq__", [](const PyAny& a, py::object b) -> bool {
        return a == PyAny(b);
    });
    variant.def(
        "get",
        [](const PyAny& self) -> py::object {
            return self.as<py::object>();
        },
        R"(
            Returns
            ----------
            get : Any
                Value of ov::Any.
        )");
    variant.def(
        "set",
        [](PyAny& self, py::object value) {
            self = PyAny(value);
        },
        R"(
            Parameters
            ----------
            set : Any
                Value to be set in ov::Any.

    )");
    variant.def_property_readonly("value", [](const PyAny& self) {
        return self.as<py::object>();
    });
}
