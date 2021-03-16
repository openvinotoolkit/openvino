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
#include <pybind11/stl_bind.h>

#include <cpp/ie_executable_network.hpp>
#include <ie_input_info.hpp>

#include "common.hpp"

#include "pyopenvino/inference_engine/ie_executable_network.hpp"
#include "pyopenvino/inference_engine/ie_input_info.hpp"

using PyConstInputsDataMap =
    std::map<std::string, std::shared_ptr<const InferenceEngine::InputInfo>>;

PYBIND11_MAKE_OPAQUE(PyConstInputsDataMap);

namespace py = pybind11;

void regclass_ExecutableNetwork(py::module m)
{
    py::class_<InferenceEngine::ExecutableNetwork,
               std::shared_ptr<InferenceEngine::ExecutableNetwork>>
        cls(m, "ExecutableNetwork");

    cls.def("create_infer_request", &InferenceEngine::ExecutableNetwork::CreateInferRequest);

    cls.def("get_exec_graph_info", &InferenceEngine::ExecutableNetwork::GetExecGraphInfo);

    cls.def("export", [](InferenceEngine::ExecutableNetwork& self,
                        const std::string& modelFileName) {
        self.Export(modelFileName);
        }, py::arg("model_file"));

    cls.def("get_config",
            [](InferenceEngine::ExecutableNetwork& self, const std::string& config_name) -> py::handle {
        return Common::parse_parameter(self.GetConfig(config_name));
    }, py::arg("config_name"));

    cls.def("get_metric",
            [](InferenceEngine::ExecutableNetwork& self, const std::string& metric_name) -> py::handle {
        return Common::parse_parameter(self.GetMetric(metric_name));
    }, py::arg("metric_name"));

    //    cls.def("get_idle_request_id", &InferenceEngine::ExecutableNetwork::CreateInferRequest);
    //
    //    cls.def("wait", &InferenceEngine::ExecutableNetwork::CreateInferRequest);

    auto py_const_inputs_data_map = py::bind_map<PyConstInputsDataMap>(m, "PyConstInputsDataMap");

    py_const_inputs_data_map.def("keys", [](PyConstInputsDataMap& self) {
        return py::make_key_iterator(self.begin(), self.end());
    });

    cls.def_property_readonly("input_info", [](InferenceEngine::ExecutableNetwork& self) {
        PyConstInputsDataMap inputs;
        const InferenceEngine::ConstInputsDataMap& inputsInfo = self.GetInputsInfo();
        for (const auto& in : inputsInfo)
        {
            inputs[in.first] = in.second;
        }
        return inputs;
    });

    cls.def_property_readonly("outputs", &InferenceEngine::ExecutableNetwork::GetOutputsInfo);

    //    cls.def_property_readonly("requests", &InferenceEngine::ExecutableNetwork::name);
}
