// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <sstream>
#include <string>

#include "Python.h"
#include "openvino/core/any.hpp"  // ov::RuntimeAttribute

namespace py = pybind11;

void regclass_graph_Any(py::module m);

class PyAny : public ov::Any {
public:
    using ov::Any::Any;
    PyAny() = default;
    PyAny(py::object object) : ov::Any(object) {}
    PyAny(PyObject* object) : ov::Any(py::reinterpret_borrow<py::object>(object)) {}
    PyAny(const ov::Any& any) : ov::Any(any) {}
};
