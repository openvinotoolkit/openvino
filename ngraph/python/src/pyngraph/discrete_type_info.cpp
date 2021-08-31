// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pyngraph/discrete_type_info.hpp"


namespace py = pybind11;

void regclass_pyngraph_DiscreteTypeInfo(py::module m) {
    py::class_<ngraph::DiscreteTypeInfo, std::shared_ptr<ngraph::DiscreteTypeInfo>> discrete_type_info(m, "DiscreteTypeInfo");
    discrete_type_info.doc() = "ngraph.impl.DiscreteTypeInfo wraps ngraph::DiscreteTypeInfo";
    discrete_type_info.def(py::init([](const std::string& _name, uint64_t version, const ngraph::DiscreteTypeInfo* parent) {
        char* name = new char[_name.size() + 1];
        std::copy(_name.begin(), _name.end(), name);
        name[_name.size()] = '\0';
        return new ngraph::DiscreteTypeInfo(name, version, parent);
    }),
    py::arg("name"),
    py::arg("version"),
    py::arg("parent") = nullptr);
    discrete_type_info.def("__lt__", [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
        return self.version < other.version || (self.version == other.version && strcmp(self.name, other.name) < 0);
    },
    py::is_operator());
    discrete_type_info.def("__le__", [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
        return self.version < other.version || (self.version == other.version && strcmp(self.name, other.name) <= 0);
    },
    py::is_operator());
    discrete_type_info.def("__gt__", [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
        return self.version < other.version || (self.version == other.version && strcmp(self.name, other.name) > 0);
    },
    py::is_operator());
    discrete_type_info.def("__ge__", [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
        return self.version < other.version || (self.version == other.version && strcmp(self.name, other.name) >= 0);
    },
    py::is_operator());
    discrete_type_info.def("__eq__", [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
        return self.version == other.version && strcmp(self.name, other.name) == 0;
    },
    py::is_operator());
    discrete_type_info.def("__ne__", [](const ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo& other) {
        return self.version != other.version || strcmp(self.name, other.name) != 0;
    },
    py::is_operator());

    discrete_type_info.def_property("name", [](const ngraph::DiscreteTypeInfo& self) {
                                                return self.name;
                                            },
                                            [](ngraph::DiscreteTypeInfo& self, const std::string& _name) {
                                                char* name = new char[_name.size() + 1];
                                                std::copy(_name.begin(), _name.end(), name);
                                                name[_name.size()] = '\0';
                                                self.name = name;
                                            });
    discrete_type_info.def_property("version", [](const ngraph::DiscreteTypeInfo& self) { return self.version; },
                                            [](ngraph::DiscreteTypeInfo& self, const uint64_t _version) {
                                                self.version = _version;
                                            });
    discrete_type_info.def_property("parent", [](const ngraph::DiscreteTypeInfo& self) { return self.parent; },
                                            [](ngraph::DiscreteTypeInfo& self, const ngraph::DiscreteTypeInfo* other) {
                                                self.parent = other;
                                            });


}
