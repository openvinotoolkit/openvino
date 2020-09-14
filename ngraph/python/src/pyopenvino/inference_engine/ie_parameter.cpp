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

#include <ie_parameter.hpp>

#include "../../../pybind11/include/pybind11/pybind11.h"
#include "pyopenvino/inference_engine/ie_parameter.hpp"

namespace py = pybind11;

void regclass_Parameter(py::module m)
{
    py::class_<InferenceEngine::Parameter,
               std::shared_ptr<InferenceEngine::Parameter>>
        cls(m, "Parameter");

    cls.def(py::init<const char*>());
    cls.def(py::init<std::string>());

    cls.def("__str__", [](){});
    cls.def("__repr__", [](){});
    cls.def("__eq__", [](){});
}
