// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

const std::shared_ptr<InferenceEngine::Blob> _convertToBlob(py::handle blob)
{
    if (py::isinstance<InferenceEngine::TBlob<float>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<double>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<double>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<int8_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int8_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<int16_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int16_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<int32_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int32_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<int64_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int64_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<uint8_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint8_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<uint16_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint16_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<uint32_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint32_t>>&>();
    }
    else if (py::isinstance<InferenceEngine::TBlob<uint64_t>>(blob))
    {
        return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint64_t>>&>();
    }
    else
    {
        // Throw error
    }
}

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def("get_blob", &InferenceEngine::InferRequest::GetBlob);
    cls.def("set_input", [](InferenceEngine::InferRequest& self, const py::dict& inputs) {
        for (auto&& input : inputs)
        {
            auto name = input.first.cast<std::string>();
            auto blob = _convertToBlob(input.second);
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_output", [](InferenceEngine::InferRequest& self, const py::dict& results) {
        for (auto&& result : results)
        {
            auto name = result.first.cast<std::string>();
            auto blob = _convertToBlob(result.second);
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_blob", [](InferenceEngine::InferRequest& self,
                           const std::string& name,
                           const InferenceEngine::TBlob<float>::Ptr& blob) {
        self.SetBlob(name, blob);
    });

    cls.def("set_blob", [](InferenceEngine::InferRequest& self,
                           const std::string& name,
                           const InferenceEngine::TBlob<float>::Ptr& blob,
                           const InferenceEngine::PreProcessInfo& info) {
        self.SetBlob(name, blob);
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
