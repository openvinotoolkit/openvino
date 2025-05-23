// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/extension.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/core/extension.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_Extension(py::module m) {
    py::class_<ov::Extension, std::shared_ptr<ov::Extension>> ext(m, "Extension", py::dynamic_attr());
    ext.doc() = "openvino.Extension provides the base interface for OpenVINO extensions.";

    ext.def("__repr__", [](const ov::Extension& self) {
        return Common::get_simple_repr(self);
    });

    ext.def(py::init<>());
}
