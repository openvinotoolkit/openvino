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

#include <pybind11/stl.h>

#include <boost/type_index.hpp>

#include <string>
#include <vector>

#include <cpp/ie_infer_request.hpp>
#include <ie_common.h>

#include "pyopenvino/inference_engine/ie_infer_request.hpp"
#include "pyopenvino/inference_engine/ie_preprocess_info.hpp"
#include "pyopenvino/inference_engine/ie_executable_network.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def("infer", &InferenceEngine::InferRequest::Infer);
    cls.def("get_blob", &InferenceEngine::InferRequest::GetBlob);
    cls.def("set_input", [](InferenceEngine::InferRequest& self, const py::dict& inputs) {
        for (auto&& input : inputs) {
            auto name = input.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob = input.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_output", [](InferenceEngine::InferRequest& self, const py::dict& results) {
        for (auto&& result : results) {
            auto name = result.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob = result.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_batch", &InferenceEngine::InferRequest::SetBatch, py::arg("size"));

    cls.def("get_perf_counts", [](InferenceEngine::InferRequest& self) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        perfMap = self.GetPerformanceCounts();
        py::dict perf_map;

        for (auto it : perfMap) {
            py::dict profile_info;
            switch (it.second.status) {
                case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                    profile_info["status"] = "EXECUTED";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                    profile_info["status"] = "NOT_RUN";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                    profile_info["status"] = "OPTIMIZED_OUT";
                    break;
                default:
                    profile_info["status"] = "UNKNOWN";
            }
            profile_info["exec_type"] = it.second.exec_type;
            profile_info["layer_type"] = it.second.layer_type;
            profile_info["cpu_time"] = it.second.cpu_uSec;
            profile_info["real_time"] = it.second.realTime_uSec;
            profile_info["execution_index"] = it.second.execution_index;
            perf_map[it.first.c_str()] = profile_info;
        }
        return perf_map;
    });

    cls.def("preprocess_info", &InferenceEngine::InferRequest::GetPreProcess, py::arg("name"));

//    cls.def_property_readonly("preprocess_info", [](InferenceEngine::InferRequest& self) {
//
//    });
//    cls.def_property_readonly("input_blobs", [](){
//
//    });
//    cls.def_property_readonly("output_blobs", [](){
//
//    });

//    cls.def("wait");
//    cls.def("set_completion_callback")
//    cls.def("async_infer",);
//    latency

//   set_blob

    //&InferenceEngine::InferRequest::SetOutput);
}
