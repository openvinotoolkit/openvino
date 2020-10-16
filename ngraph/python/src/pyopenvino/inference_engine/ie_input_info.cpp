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

#include "pyopenvino/inference_engine/ie_input_info.hpp"
#include "../../../pybind11/include/pybind11/pybind11.h"
#include "common.hpp"

namespace py = pybind11;

void regclass_InputInfo(py::module m) {
    py::class_<InferenceEngine::InputInfo, std::shared_ptr<InferenceEngine::InputInfo>> cls(m, "InputInfoPtr");

    cls.def_property("input_data", &InferenceEngine::InputInfo::getInputData,
                     &InferenceEngine::InputInfo::setInputData);

    cls.def_property("layout", [](InferenceEngine::InputInfo& self) {
            return Common::get_layout_from_enum(self.getLayout());
        }, [](InferenceEngine::InputInfo& self, const std::string& layout) {
            self.setLayout(Common::get_layout_from_string(layout));
    });

    cls.def_property("precision", [](InferenceEngine::InputInfo& self) {
        return self.getPrecision().name();
        }, [](InferenceEngine::InputInfo& self, const std::string& precision) {
        self.setPrecision(InferenceEngine::Precision::FromStr(precision));
    });

    cls.def_property_readonly("tensor_desc", &InferenceEngine::InputInfo::getTensorDesc);

    cls.def_property_readonly("name", &InferenceEngine::InputInfo::name);
}
