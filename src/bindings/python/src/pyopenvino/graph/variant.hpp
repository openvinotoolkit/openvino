// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/any.hpp"

namespace py = pybind11;

void regclass_graph_Variant(py::module m);

class PyAny : public ov::Any {
public:
    using ov::Any::Any;
    PyAny(py::object object) : ov::Any(object) {}
    PyAny(const ov::Any &any): ov::Any(any) {}
};