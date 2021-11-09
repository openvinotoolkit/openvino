// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/ie_version.hpp"

#include <ie_version.hpp>

namespace py = pybind11;

void regclass_Version(py::module m) {
    py::class_<InferenceEngine::Version> cls(m, "Version");

    cls.def_readonly("build_number", &InferenceEngine::Version::buildNumber);
    cls.def_readonly("description", &InferenceEngine::Version::description);
    cls.def_readwrite("api_version", &InferenceEngine::Version::apiVersion);

    cls.def_property_readonly("major", [](InferenceEngine::Version& self) {
        return IE_VERSION_MAJOR;
    });

    cls.def_property_readonly("minor", [](InferenceEngine::Version& self) {
        return IE_VERSION_MINOR;
    });
}