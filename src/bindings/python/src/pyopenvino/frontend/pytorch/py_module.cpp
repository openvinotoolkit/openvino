// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "decoder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_pytorch_frontend, m) {
    regclass_frontend_pytorch_decoder(m);
}
