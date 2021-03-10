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
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <boost/type_index.hpp>

#include <string>
#include <vector>

// #include <cpp/ie_infer_request.hpp> // in "pyopenvino/inference_engine/ie_infer_request.hpp"

#include "pyopenvino/inference_engine/ie_infer_request.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def("get_blob", &InferenceEngine::InferRequest::GetBlob);
    cls.def("set_input", [](InferenceEngine::InferRequest& self, const py::dict& inputs) {
        for (auto&& input : inputs)
        {
            auto name = input.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob =
                input.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_output", [](InferenceEngine::InferRequest& self, const py::dict& results) {
        for (auto&& result : results)
        {
            auto name = result.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob =
                result.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_batch", &InferenceEngine::InferRequest::SetBatch);
    cls.def("infer", &InferenceEngine::InferRequest::Infer);
    cls.def("async_infer",
            &InferenceEngine::InferRequest::StartAsync,
            py::call_guard<py::gil_scoped_release>());
    cls.def("wait",
            &InferenceEngine::InferRequest::Wait,
            py::arg("millis_timeout") = InferenceEngine::IInferRequest::WaitMode::RESULT_READY,
            py::call_guard<py::gil_scoped_acquire>());
    cls.def("set_completion_callback",
            [](InferenceEngine::InferRequest* self, py::function f_callback) {
                self->SetCompletionCallback([f_callback]() {
                    py::gil_scoped_acquire acquire;
                    f_callback();
                    py::gil_scoped_release release;
                });
            });
    //    cls.def_property_readonly("input_blobs", [](){
    //
    //    });
    //    cls.def_property_readonly("output_blobs", [](){
    //
    //    });
    //    cls.def("get_perf_counts", ); // TODO: add map of std::map<std::string,
    //    InferenceEngineProfileInfo> perfmap
    //    cls.def_property_readonly("preprocess_info", [](){
    //
    //    });
    //   cls.def("set_blob", );
}
