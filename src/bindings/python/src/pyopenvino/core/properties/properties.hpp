// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/runtime/properties.hpp"
#include "pyopenvino/core/properties/properties.hpp"

namespace py = pybind11;


template <typename T, ov::PropertyMutability mutability_, typename B = ov::util::PropertyTag>
void register_property_class(
    py::module m,
    std::string type_name) {
    py::class_<ov::Property<T, mutability_>, std::shared_ptr<ov::Property<T, mutability_>>, B> cls(
        m,
        ("PropertyType" + type_name + (mutability_ == ov::PropertyMutability::RW ? "RW" : "RO")).c_str());

    cls.def("name", [](ov::Property<T, mutability_>& self) {
        return self.name();
    });

    cls.def("__repr__", [](ov::Property<T, mutability_>& self) {
        return py::str("<class Property: \"" + std::string(self.name()) + "\">");
    });

    // Minimal set of functions that allows to pass it as dictionary entries
    cls.def("__gt__", [](const ov::Property<T, mutability_>& self, py::str& other) {
        return self.name() > py::cast<std::string>(other);
    });

    cls.def("__lt__", [](const ov::Property<T, mutability_>& self, py::str& other) {
        return self.name() < py::cast<std::string>(other);
    });

    cls.def("__lt__", [](const ov::Property<T, mutability_>& self, const py::object& other) {
        return self.name() < py::cast<std::string>(other.attr("name")());
    });
}

template <typename T>
void wrap_property_RO(py::module m, ov::Property<T, ov::PropertyMutability::RO> property, std::string name) {
    m.def(name.c_str(), [property]() {
        return property;
    });
}

template <typename T>
void wrap_property_RW(py::module m, ov::Property<T, ov::PropertyMutability::RW> property, std::string name) {
    m.def(name.c_str(), [property]() {
        return property;
    });

    m.def(name.c_str(), [property](T value) {
        // TODO: cast somehow?
        return property(value);
    });
}

void regmodule_properties(py::module m);
