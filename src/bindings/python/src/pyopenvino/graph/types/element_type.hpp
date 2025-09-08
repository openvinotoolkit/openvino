// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_graph_Type(py::module m);
void regclass_graph_Bool(py::module m);
void regclass_graph_Float32(py::module m);
void regclass_graph_Float64(py::module m);
void regclass_graph_Int8(py::module m);
// void regclass_graph_Int16(py::module m);
void regclass_graph_Int32(py::module m);
void regclass_graph_Int64(py::module m);
void regclass_graph_UInt8(py::module m);
// void regclass_graph_UInt16(py::module m);
void regclass_graph_UInt32(py::module m);
void regclass_graph_UInt64(py::module m);
void regclass_graph_BFloat16(py::module m);
