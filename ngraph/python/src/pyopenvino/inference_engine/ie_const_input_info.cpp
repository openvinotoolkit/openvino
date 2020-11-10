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


#include <ie_input_info.hpp>

#include "pyopenvino/inference_engine/ie_const_input_info.hpp"
#include "../../../pybind11/include/pybind11/pybind11.h"
#include <pybind11/stl.h>
#include "common.hpp"

namespace py = pybind11;

void regclass_ConstInputInfo(py::module m) {
//    py::class_<const InferenceEngine::InputInfo, std::shared_ptr<const InferenceEngine::InputInfo>> cls(m, "InputInfoCPtr");
//
//    cls.def_property_readonly("input_data", &InferenceEngine::InputInfo::getInputData);
//
//    cls.def_property_readonly("precision", [](InferenceEngine::InputInfo& self) {
//        return self.getPrecision().name();
//    });
//
//    cls.def_property_readonly("tensor_desc", &InferenceEngine::InputInfo::getTensorDesc);
//
//    cls.def_property_readonly("name", &InferenceEngine::InputInfo::name);
}
