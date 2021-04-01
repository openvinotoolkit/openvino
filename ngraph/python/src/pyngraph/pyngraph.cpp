//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "pyngraph/axis_set.hpp"
#include "pyngraph/axis_vector.hpp"
#include "pyngraph/coordinate.hpp"
#include "pyngraph/coordinate_diff.hpp"
#include "pyngraph/function.hpp"
#include "pyngraph/node.hpp"
#include "pyngraph/node_factory.hpp"
#include "pyngraph/node_input.hpp"
#include "pyngraph/node_output.hpp"
#if defined(NGRAPH_ONNX_IMPORT_ENABLE)
#include "pyngraph/onnx_import/onnx_import.hpp"
#endif
#include "pyngraph/dimension.hpp"
#include "pyngraph/ops/constant.hpp"
#include "pyngraph/ops/parameter.hpp"
#include "pyngraph/ops/result.hpp"
#include "pyngraph/ops/util/regmodule_pyngraph_op_util.hpp"
#include "pyngraph/partial_shape.hpp"
#include "pyngraph/passes/regmodule_pyngraph_passes.hpp"
#include "pyngraph/rt_map.hpp"
#include "pyngraph/shape.hpp"
#include "pyngraph/strides.hpp"
#include "pyngraph/types/regmodule_pyngraph_types.hpp"
#include "pyngraph/util.hpp"
#include "pyngraph/variant.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyngraph, m)
{
    m.doc() = "Package ngraph.impl that wraps nGraph's namespace ngraph";
    regclass_pyngraph_PyRTMap(m);
    regclass_pyngraph_Node(m);
    regclass_pyngraph_Input(m);
    regclass_pyngraph_Output(m);
    regclass_pyngraph_NodeFactory(m);
    regclass_pyngraph_Dimension(m); // Dimension must be registered before PartialShape
    regclass_pyngraph_PartialShape(m);
    regclass_pyngraph_Shape(m);
    regclass_pyngraph_Strides(m);
    regclass_pyngraph_CoordinateDiff(m);
    regclass_pyngraph_AxisSet(m);
    regclass_pyngraph_AxisVector(m);
    regclass_pyngraph_Coordinate(m);
    regmodule_pyngraph_types(m);
    regclass_pyngraph_Function(m);
    py::module m_op = m.def_submodule("op", "Package ngraph.impl.op that wraps ngraph::op");
    regclass_pyngraph_op_Constant(m_op);
    regclass_pyngraph_op_Parameter(m_op);
    regclass_pyngraph_op_Result(m_op);
#if defined(NGRAPH_ONNX_IMPORT_ENABLE)
    regmodule_pyngraph_onnx_import(m);
#endif
    regmodule_pyngraph_op_util(m_op);
    regmodule_pyngraph_passes(m);
    regmodule_pyngraph_util(m);
    regclass_pyngraph_Variant(m);
    regclass_pyngraph_VariantWrapper<std::string>(m, std::string("String"));
    regclass_pyngraph_VariantWrapper<int64_t>(m, std::string("Int"));
}
