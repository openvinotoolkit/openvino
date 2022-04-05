// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/utils/utils.hpp"

#include <pybind11/stl.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "Python.h"

namespace Common {
namespace utils {

py::object from_ov_any(const ov::Any& any) {
    // Check for py::object
    if (any.is<py::object>()) {
        return any.as<py::object>();
    }
    // Check for std::string
    else if (any.is<std::string>()) {
        return py::cast(any.as<std::string>().c_str());
    }
    // Check for int
    else if (any.is<int>()) {
        return py::cast(any.as<int>());
    } else if (any.is<int64_t>()) {
        return py::cast(any.as<int64_t>());
    }
    // Check for unsigned int
    else if (any.is<unsigned int>()) {
        return py::cast(any.as<unsigned int>());
    }
    // Check for float
    else if (any.is<float>()) {
        return py::cast(any.as<float>());
    } else if (any.is<double>()) {
        return py::cast(any.as<double>());
    }
    // Check for bool
    else if (any.is<bool>()) {
        return py::cast(any.as<bool>());
    }
    // Check for std::vector<std::string>
    else if (any.is<std::vector<std::string>>()) {
        return py::cast(any.as<std::vector<std::string>>());
    }
    // Check for std::vector<int>
    else if (any.is<std::vector<int>>()) {
        return py::cast(any.as<std::vector<int>>());
    }
    // Check for std::vector<int64_t>
    else if (any.is<std::vector<int64_t>>()) {
        return py::cast(any.as<std::vector<int64_t>>());
    }
    // Check for std::vector<unsigned int>
    else if (any.is<std::vector<unsigned int>>()) {
        return py::cast(any.as<std::vector<unsigned int>>());
    }
    // Check for std::vector<float>
    else if (any.is<std::vector<float>>()) {
        return py::cast(any.as<std::vector<float>>());
    }
    // Check for std::vector<double>
    else if (any.is<std::vector<double>>()) {
        return py::cast(any.as<std::vector<double>>());
    }
    // Check for std::tuple<unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int>>()) {
        return py::cast(any.as<std::tuple<unsigned int, unsigned int>>());
    }
    // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        return py::cast(any.as<std::tuple<unsigned int, unsigned int, unsigned int>>());
    }
    // Check for std::map<std::string, std::string>
    else if (any.is<std::map<std::string, std::string>>()) {
        return py::cast(any.as<std::map<std::string, std::string>>());
    }
    // Check for std::map<std::string, int>
    else if (any.is<std::map<std::string, int>>()) {
        return py::cast(any.as<std::map<std::string, int>>());
    }
    // Check for std::vector<ov::PropertyName>
    else if (any.is<std::vector<ov::PropertyName>>()) {
        auto val = any.as<std::vector<ov::PropertyName>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            std::string property_name = it;
            std::string mutability = it.is_mutable() ? "RW" : "RO";
            PyDict_SetItemString(dict, property_name.c_str(), PyUnicode_FromString(mutability.c_str()));
        }
        return py::cast<py::object>(dict);
    } else if (any.is<ov::element::Type>()) {
        return py::cast(any.as<ov::element::Type>());
    } else if (any.is<ov::hint::Priority>()) {
        return py::cast(any.as<ov::hint::Priority>());
    } else if (any.is<ov::hint::PerformanceMode>()) {
        return py::cast(any.as<ov::hint::PerformanceMode>());
    } else if (any.is<ov::log::Level>()) {
        return py::cast(any.as<ov::log::Level>());
    } else if (any.is<ov::device::Type>()) {
        return py::cast(any.as<ov::device::Type>());
    } else if (any.is<ov::streams::Num>()) {
        return py::cast(any.as<ov::streams::Num>());
    } else if (any.is<ov::Affinity>()) {
        return py::cast(any.as<ov::Affinity>());
    } else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return py::cast<py::object>((PyObject*)NULL);
    }
}

std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties) {
    std::map<std::string, ov::Any> properties_to_cpp;
    for (const auto& property : properties) {
        properties_to_cpp[property.first] = py_object_to_any(property.second);
    }
    return properties_to_cpp;
}
};  // namespace utils
};  // namespace Common
