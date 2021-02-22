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

#include <ie_data.h>

#include "pyopenvino/inference_engine/ie_data.hpp"
#include "common.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

void regclass_Data(py::module m) {
    py::class_<InferenceEngine::Data, std::shared_ptr<InferenceEngine::Data>> cls(m, "DataPtr");

    cls.def_property("layout", [](InferenceEngine::Data& self) {
        return Common::get_layout_from_enum(self.getLayout());
    }, [](InferenceEngine::Data& self, const std::string& layout) {
        self.setLayout(Common::get_layout_from_string(layout));
    });

    cls.def_property("precision", [](InferenceEngine::Data& self) {
        return self.getPrecision().name();
    }, [](InferenceEngine::Data& self, const std::string& precision) {
        self.setPrecision(InferenceEngine::Precision::FromStr(precision));
    });

    cls.def_property_readonly("shape", &InferenceEngine::Data::getDims);

    cls.def_property_readonly("name", &InferenceEngine::Data::getName);
    // cls.def_property_readonly("initialized", );
}
