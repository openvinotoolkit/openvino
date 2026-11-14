// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <string>

#include "openvino/core/type.hpp"
namespace py = pybind11;

// DiscreteTypeInfo doesn't own provided memory. Wrapper allows to avoid leaks.
class DiscreteTypeInfoWrapper : public ov::DiscreteTypeInfo {
private:
    const std::string name_str;
    const std::string version_id_str;

public:
    DiscreteTypeInfoWrapper(std::string _name_str, std::string _version_id_str)
        // Since constructor calculates hash based on name and version_id, it's needed to provide them even before
        // taking ownership of these strings.
        : DiscreteTypeInfo{_name_str.c_str(), _version_id_str.c_str(), nullptr},
          name_str{std::move(_name_str)},
          version_id_str{std::move(_version_id_str)} {
        // The reassignment is needed. The C++ standard does not guarantee that c_str() keeps the same address after a
        // std::string is move-constructed.
        name = name_str.c_str();
        version_id = version_id_str.c_str();
    }
};

void regclass_graph_DiscreteTypeInfo(py::module m);
