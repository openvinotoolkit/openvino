// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/type.hpp"
#include <string>
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


void regclass_graph_DiscreteTypeInfo(py::module m);
