// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_pyngraph_Type(py::module m);
void regclass_pyngraph_Bool(py::module m);
void regclass_pyngraph_Float32(py::module m);
void regclass_pyngraph_Float64(py::module m);
void regclass_pyngraph_Int8(py::module m);
// void regclass_pyngraph_Int16(py::module m);
void regclass_pyngraph_Int32(py::module m);
void regclass_pyngraph_Int64(py::module m);
void regclass_pyngraph_UInt8(py::module m);
// void regclass_pyngraph_UInt16(py::module m);
void regclass_pyngraph_UInt32(py::module m);
void regclass_pyngraph_UInt64(py::module m);
void regclass_pyngraph_BFloat16(py::module m);
