// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/any.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/pass/serialize.hpp"

namespace py = pybind11;

namespace Common {
namespace utils {
    py::object from_ov_any_map(const ov::AnyMap& map);

    py::object from_ov_any(const ov::Any& any);

    std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties);

    std::string convert_path_to_string(const py::object& path);

    void deprecation_warning(const std::string& function_name, const std::string& version = std::string(), const std::string& message = std::string());

    ov::Any py_object_to_any(const py::object& py_obj);

    ov::pass::Serialize::Version convert_to_version(const std::string& version);

}; // namespace utils
}; // namespace Common
