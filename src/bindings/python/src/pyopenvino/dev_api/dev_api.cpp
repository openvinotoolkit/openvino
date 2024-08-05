// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/dev_api/dev_api.hpp"

#include "../dev_api/openvino/core/bound_evaluation_util.hpp"
#include "../dev_api/openvino/core/validation_util.hpp"

namespace py = pybind11;

void regmodule_dev_api(py::module m) {
    py::module m_dev_api = m.def_submodule("dev_api", "<TODO> Package that wraps openvino dev_api");

    m_dev_api.def("evaluate_as_partial_shape", []() {
        py::print("evaluate_as_partial_shape func");
    });

    m_dev_api.def("evaluate_both_bounds", []() {
        py::print("evaluate_both_bounds func");
    });
}
