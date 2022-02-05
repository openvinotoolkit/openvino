// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "extension.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_tensorflow_frontend, m) {
    regclass_frontend_tensorflow_ConversionExtension(m);
}
