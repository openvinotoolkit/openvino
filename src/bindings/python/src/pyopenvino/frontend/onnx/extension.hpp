// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_frontend_onnx_ConversionExtension(py::module m);
void regclass_frontend_onnx_OpExtension(py::module m);
