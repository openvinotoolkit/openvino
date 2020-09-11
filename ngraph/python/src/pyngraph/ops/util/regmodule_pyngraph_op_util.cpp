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

#include "pyngraph/ops/util/regmodule_pyngraph_op_util.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_pyngraph_op_util(py::module m)
{
    py::module m_util = m.def_submodule("util", "module pyngraph.op.util");
    //    regclass_pyngraph_op_util_RequiresTensorViewArgs(m_util);
    regclass_pyngraph_op_util_OpAnnotations(m_util);
    regclass_pyngraph_op_util_ArithmeticReduction(m_util);
    //    regclass_pyngraph_op_util_BinaryElementwise(m_util);
    regclass_pyngraph_op_util_BinaryElementwiseArithmetic(m_util);
    regclass_pyngraph_op_util_BinaryElementwiseComparison(m_util);
    regclass_pyngraph_op_util_BinaryElementwiseLogical(m_util);
    //    regclass_pyngraph_op_util_UnaryElementwise(m_util);
    regclass_pyngraph_op_util_UnaryElementwiseArithmetic(m_util);
    regclass_pyngraph_op_util_IndexReduction(m_util);
}
