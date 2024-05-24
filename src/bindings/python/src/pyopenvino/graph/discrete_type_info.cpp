// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "discrete_type_info.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/core/type.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

// DiscreteTypeInfo doesn't own provided memory. Wrapper allows to avoid leaks.
class DiscreteTypeInfoWrapper : public ov::DiscreteTypeInfo {
private:
    const std::string name_str;
    const std::string version_id_str;

public:
    DiscreteTypeInfoWrapper(std::string _name_str, std::string _version_id_str)
        : DiscreteTypeInfo(nullptr, nullptr, nullptr),
          name_str(std::move(_name_str)),
          version_id_str(std::move(_version_id_str)) {
        name = name_str.c_str();
        version_id = version_id_str.c_str();
    }
};

void regclass_graph_DiscreteTypeInfo(py::module m) {
    py::class_<ov::DiscreteTypeInfo, std::shared_ptr<ov::DiscreteTypeInfo>> discrete_type_info(m, "DiscreteTypeInfo");
    discrete_type_info.doc() = "openvino.runtime.DiscreteTypeInfo wraps ov::DiscreteTypeInfo";

    discrete_type_info.def(py::init([](const std::string& name, const std::string& version_id) {
                               return std::make_shared<DiscreteTypeInfoWrapper>(name, version_id);
                           }),
                           py::arg("name"),
                           py::arg("version_id"));

    // operator overloading
    discrete_type_info.def(py::self < py::self);
    discrete_type_info.def(py::self <= py::self);
    discrete_type_info.def(py::self > py::self);
    discrete_type_info.def(py::self >= py::self);
    discrete_type_info.def(py::self == py::self);
    discrete_type_info.def(py::self != py::self);

    discrete_type_info.def_readonly("name", &ov::DiscreteTypeInfo::name);
    discrete_type_info.def_readonly("version_id", &ov::DiscreteTypeInfo::version_id);
    discrete_type_info.def_readonly("parent", &ov::DiscreteTypeInfo::parent);

    discrete_type_info.def("hash", [](const ov::DiscreteTypeInfo& self) {
        return self.hash();
    });

    discrete_type_info.def("__repr__", [](const ov::DiscreteTypeInfo& self) {
        std::string name = std::string(self.name);
        std::string version = std::string(self.version_id);
        std::string class_name = Common::get_class_name(self);
        if (self.parent != nullptr) {
            std::string parent_version = std::string(self.parent->version_id);
            std::string parent_name = self.parent->name;
            return "<" + class_name + ": " + name + " " + version + " Parent(" + parent_name + " v" + parent_version +
                   ")" + ">";
        }
        return "<" + class_name + ": " + name + " " + version + ">";
    });
}
