// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>

#include <string>

#include <ie_common.h>

#include "pyopenvino/inference_engine/common.hpp"
#include "pyopenvino/inference_engine/ie_infer_request.hpp"
#include "pyopenvino/inference_engine/ie_preprocess_info.hpp"
#include "pyopenvino/inference_engine/containers.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferRequestWrapper, std::shared_ptr<InferRequestWrapper>> cls(
        m, "InferRequest");

    cls.def("set_batch", [](InferRequestWrapper& self, const int size) {
        self._request.SetBatch(size);
    }, py::arg("size"));

    cls.def("get_blob", [](InferRequestWrapper& self, const std::string& name) {
        return self._request.GetBlob(name);
    }, py::arg("name"));

    cls.def("set_blob", [](InferRequestWrapper& self,
                           const std::string& name,
                           py::handle& blob) {
        self._request.SetBlob(name, Common::cast_to_blob(blob));
    }, py::arg("name"), py::arg("blob"));

    cls.def("set_blob", [](InferRequestWrapper& self,
                           const std::string& name,
                           py::handle& blob,
                           const InferenceEngine::PreProcessInfo& info) {
        self._request.SetBlob(name, Common::cast_to_blob(blob));
    }, py::arg("name"), py::arg("blob"), py::arg("info"));

    cls.def("set_input", [](InferRequestWrapper& self, const py::dict& inputs) {
        Common::set_request_blobs(self._request, inputs);
    }, py::arg("inputs"));

    cls.def("set_output", [](InferRequestWrapper& self, const py::dict& results) {
        Common::set_request_blobs(self._request, results);
    }, py::arg("results"));

    cls.def("_infer", [](InferRequestWrapper& self, const py::dict& inputs) {
        // Update inputs if there are any
        if (!inputs.empty()) {
            Common::set_request_blobs(self._request, inputs);
        }
        // Call Infer function
        self._startTime = Time::now();
        self._request.Infer();
        self._endTime = Time::now();
        // Get output Blobs and return
        Containers::PyResults results;
        for (auto& out : self._outputsInfo)
        {
            results[out.first] = self._request.GetBlob(out.first);
        }
        return results;
    }, py::arg("inputs"));

    cls.def(
        "_async_infer",
        [](InferRequestWrapper& self, const py::dict inputs, py::object userdata) {
            py::gil_scoped_release release;
            if (!inputs.empty())
            {
                Common::set_request_blobs(self._request, inputs);
            }
            // TODO: check for None so next async infer userdata can be updated
            // if (!userdata.empty())
            // {
            //     if (user_callback_defined)
            //     {
            //         self._request.SetCompletionCallback([self, userdata]() {
            //             // py::gil_scoped_acquire acquire;
            //             auto statusCode = const_cast<InferRequestWrapper&>(self).Wait(
            //                 InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
            //             self._request.user_callback(self, statusCode, userdata);
            //             // py::gil_scoped_release release;
            //         });
            //     }
            //     else
            //     {
            //         py::print("There is no callback function!");
            //     }
            // }
            self._startTime = Time::now();
            self._request.StartAsync();
        },
        py::arg("inputs"),
        py::arg("userdata"));

    cls.def("cancel", [](InferRequestWrapper& self) {
        self._request.Cancel();
    });

    cls.def(
        "wait",
        [](InferRequestWrapper& self, int64_t millis_timeout) {
            py::gil_scoped_release release;
            return self._request.Wait(millis_timeout);
        },
        py::arg("millis_timeout") = InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    cls.def("set_completion_callback",
            [](InferRequestWrapper& self, py::function f_callback, py::object userdata) {
                self._request.SetCompletionCallback([&self, f_callback, userdata]() {
                    self._endTime = Time::now();
                    InferenceEngine::StatusCode statusCode =
                        self._request.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
                    if (statusCode == InferenceEngine::StatusCode::RESULT_NOT_READY)
                    {
                        statusCode = InferenceEngine::StatusCode::OK;
                    }
                    // Acquire GIL, execute Python function
                    py::gil_scoped_acquire acquire;
                    f_callback(self, statusCode, userdata);
                });
            }, py::arg("f_callback"), py::arg("userdata"));

    cls.def("get_perf_counts", [](InferRequestWrapper& self) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        perfMap = self._request.GetPerformanceCounts();
        py::dict perf_map;

        for (auto it : perfMap)
        {
            py::dict profile_info;
            switch (it.second.status)
            {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                profile_info["status"] = "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                profile_info["status"] = "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                profile_info["status"] = "OPTIMIZED_OUT";
                break;
            default: profile_info["status"] = "UNKNOWN";
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

    cls.def(
        "preprocess_info",
        [](InferRequestWrapper& self, const std::string& name) {
            return self._request.GetPreProcess(name);
        },
        py::arg("name"));

    //    cls.def_property_readonly("preprocess_info", [](InferRequestWrapper& self) {
    //
    //    });

    cls.def_property_readonly("input_blobs", [](InferRequestWrapper& self) {
        Containers::PyResults input_blobs;
        for (auto& in : self._inputsInfo)
        {
            input_blobs[in.first] = self._request.GetBlob(in.first);
        }
        return input_blobs;
    });

    cls.def_property_readonly("output_blobs", [](InferRequestWrapper& self) {
        Containers::PyResults output_blobs;
        for (auto& out : self._outputsInfo)
        {
            output_blobs[out.first] = self._request.GetBlob(out.first);
        }
        return output_blobs;
    });

    cls.def_property_readonly("latency", [](InferRequestWrapper& self) {
        return self.getLatency();
    });
}
