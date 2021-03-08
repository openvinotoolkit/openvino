//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ie_version.hpp>

#include "pyopenvino/inference_engine/ie_version.hpp"

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