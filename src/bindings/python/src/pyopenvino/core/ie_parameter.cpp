// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/ie_parameter.hpp"

#include <ie_parameter.hpp>

namespace py = pybind11;

void regclass_Parameter(py::module m) {
    py::class_<InferenceEngine::Parameter, std::shared_ptr<InferenceEngine::Parameter>> cls(m, "Parameter");

    cls.def(py::init<const char*>());
    cls.def(py::init<std::string>());
}
