// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_tensorflow_FrontEnd(py::module m);
void regclass_tensorflow_NodeContext(py::module m);
void regclass_tensorflow_ConversionExtension(py::module m);
