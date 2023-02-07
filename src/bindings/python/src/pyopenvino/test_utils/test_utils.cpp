// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(test_utils_api, m) {
    m.def(
        "compare_functions",
        [](const ov::Model& lhs,
           const ov::Model& rhs,
           bool compare_const_values,
           bool compare_names,
           bool compare_runtime_keys,
           bool compare_precisions,
           bool compare_attributes,
           bool compare_tensor_names,
           bool compare_accuracy,
           bool compare_subgraph_descriptors,
           bool compare_consumers_count) {
            const auto lhs_ptr = std::const_pointer_cast<ov::Model>(lhs.shared_from_this());
            const auto rhs_ptr = std::const_pointer_cast<ov::Model>(rhs.shared_from_this());

            auto fc = FunctionsComparator::no_default().enable(FunctionsComparator::NODES);
            if (compare_const_values)
                fc.enable(FunctionsComparator::CONST_VALUES);
            if (compare_names)
                fc.enable(FunctionsComparator::NAMES);
            if (compare_runtime_keys)
                fc.enable(FunctionsComparator::RUNTIME_KEYS);
            if (compare_precisions)
                fc.enable(FunctionsComparator::PRECISIONS);
            if (compare_attributes)
                fc.enable(FunctionsComparator::ATTRIBUTES);
            if (compare_tensor_names)
                fc.enable(FunctionsComparator::TENSOR_NAMES);
            if (compare_accuracy)
                fc.enable(FunctionsComparator::ACCURACY);
            if (compare_subgraph_descriptors)
                fc.enable(FunctionsComparator::SUBGRAPH_DESCRIPTORS);
            if (compare_consumers_count)
                fc.enable(FunctionsComparator::CONSUMERS_COUNT);

            const auto results = fc.compare(lhs_ptr, rhs_ptr);
            return std::make_pair(results.valid, results.message);
        },
        py::arg("lhs"),
        py::arg("rhs"),
        py::arg("compare_const_values") = true,
        py::arg("compare_names") = false,
        py::arg("compare_runtime_keys") = false,
        py::arg("compare_precisions") = true,
        py::arg("compare_attributes") = true,
        py::arg("compare_tensor_names") = true,
        py::arg("compare_accuracy") = false,
        py::arg("compare_subgraph_descriptors") = true,
        py::arg("compare_consumers_count") = true);
}
