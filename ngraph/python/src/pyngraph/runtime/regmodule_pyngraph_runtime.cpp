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

#include "pyngraph/runtime/regmodule_pyngraph_runtime.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_pyngraph_runtime(py::module m)
{
    py::module m_runtime =
        m.def_submodule("runtime", "Package ngraph.impl.runtime wraps ngraph::runtime");
    regclass_pyngraph_runtime_Tensor(m_runtime);
    regclass_pyngraph_runtime_Backend(m_runtime);
    regclass_pyngraph_runtime_Executable(m_runtime);
}
