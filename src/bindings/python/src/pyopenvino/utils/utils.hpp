// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/any.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/frontend/decoder.hpp"

namespace py = pybind11;

namespace Common {
namespace utils {
py::object from_ov_any(const ov::Any& any);

std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties);

std::string convert_path_to_string(const py::object& path);

void deprecation_warning(const std::string& function_name,
                         const std::string& version = std::string(),
                         const std::string& message = std::string());
};  // namespace utils
};  // namespace Common

inline ov::Any py_object_to_any(const py::object& py_obj) {
    // TODO: Investigate if there is a better alternative for converting any registered pybind11 type
    // Just listing all known types here looks a double work as we have already registed a lot of OV types
    // in other pybind11 definitions.
    // Another option is to not unpack pybind object until ov::Any is casted.

    // Python types
    if (py::isinstance<py::str>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::bool_>(py_obj)) {
        return py_obj.cast<bool>();
    } else if (py::isinstance<py::float_>(py_obj)) {
        return py_obj.cast<double>();
    } else if (py::isinstance<py::int_>(py_obj)) {
        return py_obj.cast<int64_t>();
    } else if (py::isinstance<py::list>(py_obj)) {
        auto _list = py_obj.cast<py::list>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL };
        PY_TYPE detected_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _list) {
            auto check_type = [&](PY_TYPE type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_ASSERT("Incorrect attribute. Mixed types in the list are not allowed.");
            };
            if (py::isinstance<py::str>(it)) {
                check_type(PY_TYPE::STR);
            } else if (py::isinstance<py::int_>(it)) {
                check_type(PY_TYPE::INT);
            } else if (py::isinstance<py::float_>(it)) {
                check_type(PY_TYPE::FLOAT);
            } else if (py::isinstance<py::bool_>(it)) {
                check_type(PY_TYPE::BOOL);
            }
        }

        // In case of empty vector works like with vector of strings
        if (_list.empty())
            return _list.cast<std::vector<std::string>>();

        switch (detected_type) {
        case PY_TYPE::STR:
            return _list.cast<std::vector<std::string>>();
        case PY_TYPE::FLOAT:
            return _list.cast<std::vector<double>>();
        case PY_TYPE::INT:
            return _list.cast<std::vector<int64_t>>();
        case PY_TYPE::BOOL:
            return _list.cast<std::vector<bool>>();
        default:
            OPENVINO_ASSERT(false, "Unsupported attribute type.");
        }
        // OV types
    } else if (py::isinstance<ov::Any>(py_obj)) {
        return py::cast<ov::Any>(py_obj);
    } else if (py::isinstance<ov::element::Type>(py_obj)) {
        return py::cast<ov::element::Type>(py_obj);
    } else if (py::isinstance<ov::hint::Priority>(py_obj)) {
        return py::cast<ov::hint::Priority>(py_obj);
    } else if (py::isinstance<ov::hint::PerformanceMode>(py_obj)) {
        return py::cast<ov::hint::PerformanceMode>(py_obj);
    } else if (py::isinstance<ov::log::Level>(py_obj)) {
        return py::cast<ov::log::Level>(py_obj);
    } else if (py::isinstance<ov::device::Type>(py_obj)) {
        return py::cast<ov::device::Type>(py_obj);
    } else if (py::isinstance<ov::streams::Num>(py_obj)) {
        return py::cast<ov::streams::Num>(py_obj);
    } else if (py::isinstance<ov::Affinity>(py_obj)) {
        return py::cast<ov::Affinity>(py_obj);
        // FrontEnd Decoder
    } else if (py::isinstance<ov::frontend::IDecoder>(py_obj)) {
        return py::cast<std::shared_ptr<ov::frontend::IDecoder>>(py_obj);
        // Custom FrontEnd Types
    } else if (py::isinstance<ov::frontend::type::Tensor>(py_obj)) {
        return py::cast<ov::frontend::type::Tensor>(py_obj);
    } else if (py::isinstance<ov::frontend::type::List>(py_obj)) {
        return py::cast<ov::frontend::type::List>(py_obj);
    }
    // If there is no match fallback to py::object
    else if (py::isinstance<py::object>(py_obj)) {
        return py_obj;
    }
    OPENVINO_ASSERT(false, "Unsupported attribute type.");
}
