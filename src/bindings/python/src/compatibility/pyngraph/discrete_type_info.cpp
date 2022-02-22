// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyngraph/discrete_type_info.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "ngraph/type.hpp"

namespace py = pybind11;

void regclass_pyngraph_DiscreteTypeInfo(py::module m) {
    py::class_<ngraph::DiscreteTypeInfo, std::shared_ptr<ngraph::DiscreteTypeInfo>> discrete_type_info(
        m,
        "DiscreteTypeInfo",
        py::module_local());
    discrete_type_info.doc() = "ngraph.impl.DiscreteTypeInfo wraps ngraph::DiscreteTypeInfo";

    // operator overloading
    discrete_type_info.def(py::self < py::self);
    discrete_type_info.def(py::self <= py::self);
    discrete_type_info.def(py::self > py::self);
    discrete_type_info.def(py::self >= py::self);
    discrete_type_info.def(py::self == py::self);
    discrete_type_info.def(py::self != py::self);

    discrete_type_info.def_readonly("name", &ngraph::DiscreteTypeInfo::name);
    discrete_type_info.def_readonly("version", &ngraph::DiscreteTypeInfo::version);
    discrete_type_info.def_readonly("parent", &ngraph::DiscreteTypeInfo::parent);

    discrete_type_info.def("__repr__", [](const ngraph::DiscreteTypeInfo& self) {
        std::string name = std::string(self.name);
        std::string version = std::to_string(self.version);
        if (self.parent != nullptr) {
            std::string parent_version = std::to_string(self.parent->version);
            std::string parent_name = self.parent->name;
            return "<DiscreteTypeInfo: " + name + " v" + version + " Parent(" + parent_name + " v" + parent_version +
                   ")" + ">";
        }
        return "<DiscreteTypeInfo: " + name + " v" + version + ">";
    });
}
