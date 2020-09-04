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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <boost/type_index.hpp>

#include <string>
#include <vector>

#include <cpp/ie_infer_request.hpp>

#include "../../../pybind11/include/pybind11/pybind11.h"
#include "pyopenvino/inference_engine/ie_executable_network.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def("infer", &InferenceEngine::InferRequest::Infer);
    cls.def("get_blob", &InferenceEngine::InferRequest::GetBlob);
    cls.def("set_input", [](InferenceEngine::InferRequest& self, const py::dict& inputs) {
        for (auto&& input : inputs)
        {
            auto name = input.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob = input.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_output", [](InferenceEngine::InferRequest& self, const py::dict& results) {
        for (auto&& result : results)
        {
            auto name = result.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob = result.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });

    //&InferenceEngine::InferRequest::SetOutput);
}
