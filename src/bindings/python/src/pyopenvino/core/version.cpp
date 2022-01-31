// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/version.hpp"

#include <pybind11/stl.h>

namespace py = pybind11;

void regclass_Version(py::module m) {
    py::class_<ov::Version> cls(m, "Version");
    cls.doc() =
        "openvino.runtime.Version represents version information that describes plugins and the OpenVINO library.";

    cls.def_readonly("build_number",
                     &ov::Version::buildNumber,
                     R"(
                        Returns
                        ----------
                        build_number : str
                            String with build number.
                     )");

    cls.def_readonly("description",
                     &ov::Version::description,
                     R"(
                        Returns
                        ----------
                        description : str
                            Description string.
                     )");

    cls.def_property_readonly(
        "major",
        [](ov::Version& self) {
            return OPENVINO_VERSION_MAJOR;
        },
        R"(
            Returns
            ----------
            major : int
                OpenVINO's major version.
        )");

    cls.def_property_readonly(
        "minor",
        [](ov::Version& self) {
            return OPENVINO_VERSION_MINOR;
        },
        R"(
        Returns
        ----------
        minor : int
            OpenVINO's minor version.
    )");
}
