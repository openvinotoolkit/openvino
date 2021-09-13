// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_version.hpp>

#include "pyopenvino/core/ie_version.hpp"

namespace py = pybind11;

void regclass_Version(py::module m) {
    py::class_<InferenceEngine::Version> cls(m, "Version");

    cls.def_readonly("build_number", &InferenceEngine::Version::buildNumber);
    cls.def_readonly("description", &InferenceEngine::Version::description);
    cls.def_readwrite("api_version", &InferenceEngine::Version::apiVersion);

    using ApiVersionType = decltype(InferenceEngine::Version::apiVersion);
    py::class_<ApiVersionType> strct(m, "apiVersion");
    strct.def_readwrite("major", &ApiVersionType::major);
    strct.def_readwrite("minor", &ApiVersionType::minor);

    cls.def_property_readonly("major", [](InferenceEngine::Version& self){
       return self.apiVersion.major;
    });

    cls.def_property_readonly("minor", [](InferenceEngine::Version& self){
        return self.apiVersion.minor;
    });
}