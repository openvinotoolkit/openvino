// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(test_utils, m) {
    void regmodule_test_utils(py::module m);
}
