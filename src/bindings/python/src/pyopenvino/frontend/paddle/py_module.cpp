// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "extension.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_paddle_frontend, m) {
    regclass_frontend_paddle_ConversionExtension(m);
}
