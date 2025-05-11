// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "common_test_utils/ov_test_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(test_utils_api, m) {
    m.def(
        "compare_functions",
        [](const ov::Model& lhs, const ov::Model& rhs, bool compare_tensor_names) {
            const auto lhs_ptr = std::const_pointer_cast<ov::Model>(lhs.shared_from_this());
            const auto rhs_ptr = std::const_pointer_cast<ov::Model>(rhs.shared_from_this());

            auto fc = FunctionsComparator::with_default()
                          .enable(FunctionsComparator::ATTRIBUTES)
                          .enable(FunctionsComparator::CONST_VALUES);

            if (!compare_tensor_names)
                fc.disable(FunctionsComparator::TENSOR_NAMES);
            const auto results = fc.compare(lhs_ptr, rhs_ptr);
            return std::make_pair(results.valid, results.message);
        },
        py::arg("lhs"),
        py::arg("rhs"),
        py::arg("compare_tensor_names") = true);
}
