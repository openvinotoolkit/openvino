// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "pyngraph/frontend/frontend.hpp"
#include "pyngraph/frontend/frontend_manager.hpp"
#include "pyngraph/frontend/inputmodel.hpp"
#include "pyngraph/frontend/place.hpp"
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
    regmodule_pyngraph_types(m);
    regclass_pyngraph_Dimension(m); // Dimension must be registered before PartialShape
    regclass_pyngraph_Shape(m);
    regclass_pyngraph_PartialShape(m);
    regclass_pyngraph_Node(m);
    regclass_pyngraph_Place(m);
    regclass_pyngraph_InitializationFailureFrontEnd(m);
    regclass_pyngraph_GeneralFailureFrontEnd(m);
    regclass_pyngraph_OpConversionFailureFrontEnd(m);
    regclass_pyngraph_OpValidationFailureFrontEnd(m);
    regclass_pyngraph_NotImplementedFailureFrontEnd(m);
    regclass_pyngraph_FrontEndManager(m);
    regclass_pyngraph_FrontEnd(m);
    regclass_pyngraph_InputModel(m);
    regclass_pyngraph_Input(m);
    regclass_pyngraph_Output(m);
    regclass_pyngraph_NodeFactory(m);
    regclass_pyngraph_Strides(m);
    regclass_pyngraph_CoordinateDiff(m);
    regclass_pyngraph_AxisSet(m);
    regclass_pyngraph_AxisVector(m);
    regclass_pyngraph_Coordinate(m);
    py::module m_op = m.def_submodule("op", "Package ngraph.impl.op that wraps ngraph::op");
    regclass_pyngraph_op_Constant(m_op);
    regclass_pyngraph_op_Parameter(m_op);
    regclass_pyngraph_op_Result(m_op);
#if defined(NGRAPH_ONNX_IMPORT_ENABLE)
    regmodule_pyngraph_onnx_import(m);
#endif
    regmodule_pyngraph_op_util(m_op);
    regclass_pyngraph_Function(m);
    regmodule_pyngraph_passes(m);
    regmodule_pyngraph_util(m);
    regclass_pyngraph_Variant(m);
    regclass_pyngraph_VariantWrapper<std::string>(m, std::string("String"));
    regclass_pyngraph_VariantWrapper<int64_t>(m, std::string("Int"));
}
