// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.hpp"

#include <pybind11/pybind11.h>

#include <string>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace py = pybind11;

void regmodule_test_utils(py::module m) {
    py::module m_test_utils = m.def_submodule("test_utils_api", "test_utils module");

    m_test_utils.def(
        "compare_functions",
        [](std::shared_ptr<ov::Function> lhs, std::shared_ptr<ov::Function> rhs) {
            return compare_functions(lhs, rhs, true, true, false, true, true);
        },
        py::arg("lhs"),
        py::arg("rhs"));
}
