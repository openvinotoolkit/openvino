// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/ie_executable_network.hpp"

#include <cpp/ie_executable_network.hpp>
#include <ie_input_info.hpp>

#include "common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/ie_infer_request.hpp"
#include "pyopenvino/core/ie_input_info.hpp"

namespace py = pybind11;

void regclass_ExecutableNetwork(py::module m) {
    py::class_<InferenceEngine::ExecutableNetwork, std::shared_ptr<InferenceEngine::ExecutableNetwork>> cls(
        m,
        "ExecutableNetwork");

    cls.def("create_infer_request", [](InferenceEngine::ExecutableNetwork& self) {
        auto request = InferRequestWrapper(self.CreateInferRequest());
        // Get Inputs and Outputs info from executable network
        request._inputsInfo = self.GetInputsInfo();
        request._outputsInfo = self.GetOutputsInfo();
        // request.user_callback_defined = false;
        return request;
    });

    cls.def(
        "_infer",
        [](InferenceEngine::ExecutableNetwork& self, const py::dict& inputs) {
            // Create temporary InferRequest
            auto request = self.CreateInferRequest();
            // Update inputs if there are any
            if (!inputs.empty()) {
                Common::set_request_blobs(request, inputs);  //, self.GetInputsInfo());
            }
            // Call Infer function
            request.Infer();
            // Get output Blobs and return
            Containers::PyResults results;
            InferenceEngine::ConstOutputsDataMap outputsInfo = self.GetOutputsInfo();
            for (auto& out : outputsInfo) {
                results[out.first] = request.GetBlob(out.first);
            }
            return results;
        },
        py::arg("inputs"));

    cls.def("get_exec_graph_info", &InferenceEngine::ExecutableNetwork::GetExecGraphInfo);

    cls.def(
        "export",
        [](InferenceEngine::ExecutableNetwork& self, const std::string& modelFileName) {
            self.Export(modelFileName);
        },
        py::arg("model_file"));

    cls.def(
        "get_config",
        [](InferenceEngine::ExecutableNetwork& self, const std::string& config_name) -> py::handle {
            return Common::parse_parameter(self.GetConfig(config_name));
        },
        py::arg("config_name"));

    cls.def(
        "get_metric",
        [](InferenceEngine::ExecutableNetwork& self, const std::string& metric_name) -> py::handle {
            return Common::parse_parameter(self.GetMetric(metric_name));
        },
        py::arg("metric_name"));

    cls.def_property_readonly("input_info", [](InferenceEngine::ExecutableNetwork& self) {
        Containers::PyConstInputsDataMap inputs;
        const InferenceEngine::ConstInputsDataMap& inputsInfo = self.GetInputsInfo();
        for (const auto& in : inputsInfo) {
            inputs[in.first] = in.second;
        }
        return inputs;
    });

    cls.def_property_readonly("output_info", [](InferenceEngine::ExecutableNetwork& self) {
        Containers::PyOutputsDataMap outputs;
        InferenceEngine::ConstOutputsDataMap outputsInfo = self.GetOutputsInfo();
        for (auto& out : outputsInfo) {
            outputs[out.first] = out.second;
        }
        return outputs;
    });
}
