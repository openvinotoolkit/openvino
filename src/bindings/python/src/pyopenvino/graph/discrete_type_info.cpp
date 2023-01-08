// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "discrete_type_info.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/core/type.hpp"

namespace py = pybind11;

void regclass_graph_DiscreteTypeInfo(py::module m) {
    py::class_<ov::DiscreteTypeInfo, std::shared_ptr<ov::DiscreteTypeInfo>> discrete_type_info(m, "DiscreteTypeInfo");
    discrete_type_info.doc() = "openvino.runtime.DiscreteTypeInfo wraps ov::DiscreteTypeInfo";

    // operator overloading
    discrete_type_info.def(py::self < py::self);
    discrete_type_info.def(py::self <= py::self);
    discrete_type_info.def(py::self > py::self);
    discrete_type_info.def(py::self >= py::self);
    discrete_type_info.def(py::self == py::self);
    discrete_type_info.def(py::self != py::self);

    discrete_type_info.def_readonly("name", &ov::DiscreteTypeInfo::name);
    discrete_type_info.def_readonly("version", &ov::DiscreteTypeInfo::version);
    discrete_type_info.def_readonly("version_id", &ov::DiscreteTypeInfo::version_id);
    discrete_type_info.def_readonly("parent", &ov::DiscreteTypeInfo::parent);

    discrete_type_info.def("get_version", &ov::DiscreteTypeInfo::get_version);
    discrete_type_info.def("hash", [](const ov::DiscreteTypeInfo& self) {
        return self.hash();
    });

    discrete_type_info.def("__repr__", [](const ov::DiscreteTypeInfo& self) {
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
