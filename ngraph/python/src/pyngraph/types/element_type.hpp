//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
