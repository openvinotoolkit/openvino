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
#include "openvino/runtime/properties.hpp"

namespace Common {
namespace utils {

py::object from_ov_any(const ov::Any& any) {
    // Check for py::object
    if (any.is<py::object>()) {
        return any.as<py::object>();
    }
    // Check for std::string
    else if (any.is<std::string>()) {
        return py::cast<py::object>(PyUnicode_FromString(any.as<std::string>().c_str()));
    }
    // Check for int
    else if (any.is<int>()) {
        return py::cast<py::object>(PyLong_FromLong(any.as<int>()));
    } else if (any.is<int64_t>()) {
        return py::cast<py::object>(PyLong_FromLong(any.as<int64_t>()));
    }
    // Check for unsigned int
    else if (any.is<unsigned int>()) {
        return py::cast<py::object>(PyLong_FromLong(any.as<unsigned int>()));
    }
    // Check for float
    else if (any.is<float>()) {
        return py::cast<py::object>(PyFloat_FromDouble(any.as<float>()));
    } else if (any.is<double>()) {
        return py::cast<py::object>(PyFloat_FromDouble(any.as<double>()));
    }
    // Check for bool
    else if (any.is<bool>()) {
        return py::cast<py::object>(any.as<bool>() ? Py_True : Py_False);
    }
    // Check for std::vector<std::string>
    else if (any.is<std::vector<std::string>>()) {
        auto val = any.as<std::vector<std::string>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyObject* str_val = PyUnicode_FromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return py::cast<py::object>(list);
    }
    // Check for std::vector<int>
    else if (any.is<std::vector<int>>()) {
        auto val = any.as<std::vector<int>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return py::cast<py::object>(list);
    }
    // Check for std::vector<int64_t>
    else if (any.is<std::vector<int64_t>>()) {
        auto val = any.as<std::vector<int64_t>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return py::cast<py::object>(list);
    }
    // Check for std::vector<unsigned int>
    else if (any.is<std::vector<unsigned int>>()) {
        auto val = any.as<std::vector<unsigned int>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return py::cast<py::object>(list);
    }
    // Check for std::vector<float>
    else if (any.is<std::vector<float>>()) {
        auto val = any.as<std::vector<float>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyFloat_FromDouble((double)it));
        }
        return py::cast<py::object>(list);
    }
    // Check for std::vector<double>
    else if (any.is<std::vector<double>>()) {
        auto val = any.as<std::vector<double>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyFloat_FromDouble(it));
        }
        return py::cast<py::object>(list);
    }
    // Check for std::tuple<unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int>>()) {
        auto val = any.as<std::tuple<unsigned int, unsigned int>>();
        PyObject* tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        return py::cast<py::object>(tuple);
    }
    // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        auto val = any.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
        PyObject* tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long)std::get<2>(val)));
        return py::cast<py::object>(tuple);
    }
    // Check for std::map<std::string, std::string>
    else if (any.is<std::map<std::string, std::string>>()) {
        auto val = any.as<std::map<std::string, std::string>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return py::cast<py::object>(dict);
    }
    // Check for std::map<std::string, int>
    else if (any.is<std::map<std::string, int>>()) {
        auto val = any.as<std::map<std::string, int>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long)it.second));
        }
        return py::cast<py::object>(dict);
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
    } else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return py::cast<py::object>((PyObject*)NULL);
    }
}
};  // namespace utils
};  // namespace Common