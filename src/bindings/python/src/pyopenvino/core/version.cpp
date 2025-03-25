// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/version.hpp"

#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/version.hpp"

namespace py = pybind11;

void regclass_Version(py::module m) {
    py::class_<ov::Version> cls(m, "Version");
    cls.doc() = "openvino.Version represents version information that describes plugins and the OpenVINO library.";

    cls.def("__repr__", [](const ov::Version& self) {
        return "<" + Common::get_class_name(self) + ": " + std::string(self.buildNumber) + " " + self.description + ">";
    });

    cls.def_readonly("build_number",
                     &ov::Version::buildNumber,
                     R"(
                        :return: String with build number.
                        :rtype: str
                     )");

    cls.def_readonly("description",
                     &ov::Version::description,
                     R"(
                        :return: Description string.
                        :rtype: str
                     )");

    cls.def_property_readonly(
        "major",
        [](ov::Version& self) {
            return OPENVINO_VERSION_MAJOR;
        },
        R"(
            :return: OpenVINO's major version.
            :rtype: int
        )");

    cls.def_property_readonly(
        "minor",
        [](ov::Version& self) {
            return OPENVINO_VERSION_MINOR;
        },
        R"(
            :return: OpenVINO's minor version.
            :rtype: int
        )");

    cls.def_property_readonly(
        "patch",
        [](ov::Version& self) {
            return OPENVINO_VERSION_PATCH;
        },
        R"(
            :return: OpenVINO's version patch.
            :rtype: int
        )");
}
