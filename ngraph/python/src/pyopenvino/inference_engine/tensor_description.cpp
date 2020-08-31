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

#include <ie_layouts.h>
#include <ie_common.h>
#include <ie_precision.hpp>

#include "pyopenvino/inference_engine/ie_network.hpp"
#include "../../../pybind11/include/pybind11/pybind11.h"
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace InferenceEngine;


Layout layout_from_string(std::string layout) {
    std::map<std::string, Layout> layout_str_to_enum = {
            {"ANY", Layout::ANY},
            {"NHWC", Layout::NHWC},
            {"NCHW", Layout::NCHW},
            {"NCDHW", Layout::NCDHW},
            {"NDHWC", Layout::NDHWC},
            {"OIHW", Layout::OIHW},
            {"GOIHW", Layout::GOIHW},
            {"OIDHW", Layout::OIDHW},
            {"GOIDHW", Layout::GOIDHW},
            {"SCALAR", Layout::SCALAR},
            {"C", Layout::C},
            {"CHW", Layout::CHW},
            {"HW", Layout::HW},
            {"NC", Layout::NC},
            {"CN", Layout::CN},
            {"BLOCKED", Layout::BLOCKED}
    };
    return layout_str_to_enum[layout];
}


void regclass_TensorDecription(py::module m)
{
    py::class_<TensorDesc, std::shared_ptr<TensorDesc>> cls(m, "TensorDesc");
    cls.def(py::init<const Precision&, const SizeVector&, Layout>());
    cls.def(py::init([](const std::string& precision, const SizeVector& dims, const std::string& layout) {
        return TensorDesc(Precision::FromStr(precision), dims, layout_from_string(layout));
    }));
}
