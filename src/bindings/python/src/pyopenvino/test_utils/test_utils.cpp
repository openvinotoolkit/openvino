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
                const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);

                const auto results = fc.compare(lhs, rhs);
                return std::make_pair(results.valid, results.message);
            },
            py::arg("lhs"),
            py::arg("rhs"));
}
