// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(test_utils_api, m) {
    m.def("compare_functions",
            [](std::shared_ptr<ov::Function> lhs, std::shared_ptr<ov::Function> rhs) {
                return compare_functions(lhs, rhs, true, true, false, true, true);
            },
            py::arg("lhs"),
            py::arg("rhs"));
}
