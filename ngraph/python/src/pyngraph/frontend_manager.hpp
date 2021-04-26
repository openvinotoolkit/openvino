// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_pyngraph_FrontEndManager(py::module m);
void regclass_pyngraph_FrontEnd(py::module m);
void regclass_pyngraph_InputModel(py::module m);
void regclass_pyngraph_FEC(py::module m);
void regclass_pyngraph_Place(py::module m);
