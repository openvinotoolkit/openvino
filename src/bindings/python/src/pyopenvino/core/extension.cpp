// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/manager.hpp"

namespace py = pybind11;

void regclass_Extension(py::module m) {
    py::class_<ov::Extension, std::shared_ptr<ov::Extension>> ext(m, "Extension", py::dynamic_attr());
    ext.doc() = "openvino.runtime.Extension provides the base interface for OpenVINO extensions.";

    ext.def(py::init<>());
}
