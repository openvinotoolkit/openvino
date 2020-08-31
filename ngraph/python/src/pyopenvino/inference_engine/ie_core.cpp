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

#include <ie_core.hpp>

#include "pyopenvino/inference_engine/ie_core.hpp"
#include "../../../pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);

}


void regclass_IECore(py::module m)
{
    py::class_<InferenceEngine::Core, std::shared_ptr<InferenceEngine::Core>> cls(m, "IECore");
    cls.def(py::init());
    cls.def(py::init<const std::string&>());

    cls.def("set_config", [](InferenceEngine::Core& self, py::dict config, py::kwargs kwargs) {
        std::map <std::string, std::string> config_map;
        for (auto item : config) {
            config_map[to_string(item.first)] = to_string(item.second);
        }
        self.SetConfig(config_map, to_string(kwargs["device_name"]));
    });

    cls.def("load_network", [](InferenceEngine::Core& self, InferenceEngine::CNNNetwork network, std::string device_name) {
        return self.LoadNetwork(network, device_name);
    });

}
