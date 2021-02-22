//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "../../../pybind11/include/pybind11/pybind11.h"
#include "pyopenvino/inference_engine/ie_infer_request.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def(py::init<InferenceEngine::ExecutableNetwork, size_t>());

    cls.def("infer", []() {

    });
    cls.def("start". []() {

    });

    cls.def("set_infer_callback", []() {
    
    });
}
