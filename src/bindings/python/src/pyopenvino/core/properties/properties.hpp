// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/runtime/properties.hpp"
#include "pyopenvino/core/properties/properties.hpp"

namespace py = pybind11;

template <typename B, typename T, ov::PropertyMutability mutability_>
py::class_<ov::Property<T, mutability_>, std::shared_ptr<ov::Property<T, mutability_>>, B> wrap_property(
    py::module m,
    std::string type_name) {
    py::class_<ov::Property<T, mutability_>, std::shared_ptr<ov::Property<T, mutability_>>, B> cls(
        m,
        ("PropertyType" + type_name).c_str());

    cls.def("name", [](ov::Property<T, mutability_>& self) {
        return self.name();
    });

    cls.def("__repr__", [](ov::Property<T, mutability_>& self) {
        return py::str("<class Property[" + std::string(self.name()) + "]>");
    });

    return std::move(cls);  // return allows us to update class later
}

void regmodule_properties(py::module m);
