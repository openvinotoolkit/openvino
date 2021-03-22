// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
