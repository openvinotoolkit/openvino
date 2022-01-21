// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/version.hpp"

#include <pybind11/stl.h>

namespace py = pybind11;

void regclass_Version(py::module m) {
    py::class_<ov::Version> cls(m, "Version");

    cls.def_readonly("build_number", &ov::Version::buildNumber);
    cls.def_readonly("description", &ov::Version::description);

    cls.def_property_readonly("major", [](ov::Version& self) {
        return OPENVINO_VERSION_MAJOR;
    });

    cls.def_property_readonly("minor", [](ov::Version& self) {
        return OPENVINO_VERSION_MINOR;
    });
}
