// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "extension.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_onnx_frontend, m) {
    regclass_frontend_onnx_ConversionExtension(m);
    regclass_frontend_onnx_OpExtension(m);
}
