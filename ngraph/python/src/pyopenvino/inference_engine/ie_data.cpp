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

#include "ie_data.hpp"
#include "pyopenvino/inference_engine/ie_input_info.hpp"
#include "../../../pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;

void regclass_Data(py::module m) {
//    py::class_<InferenceEngine::Data, std::shared_ptr<InferenceEngine::Data>> cls(m, "DataPtr");
//
//    cls.def_property("layout", &InferenceEngine::Data::getLayout,
//                     &InferenceEngine::CNNNetwork::setLayout);
//
//    cls.def_property("precision", &InferenceEngine::Data::getPrecision,
//                     &InferenceEngine::CNNNetwork::setPrecision);
//
//    cls.def_property_readonly("shape", &InferenceEngine::Data::getDims);
}