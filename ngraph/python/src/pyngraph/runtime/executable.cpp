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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "pyngraph/runtime/executable.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_Executable(py::module m)
{
    py::class_<ngraph::runtime::Executable, std::shared_ptr<ngraph::runtime::Executable>>
        executable(m, "Executable");
    executable.doc() = "ngraph.impl.runtime.Executable wraps ngraph::runtime::Executable";
    executable.def("call",
                   (bool (ngraph::runtime::Executable::*)(
                       const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>&,
                       const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>&)) &
                       ngraph::runtime::Executable::call);
    executable.def(
        "get_performance_data",
        (std::vector<ngraph::runtime::PerformanceCounter>(ngraph::runtime::Executable::*)()) &
            ngraph::runtime::Executable::get_performance_data);
}
