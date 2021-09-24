// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/axis_set.hpp"
#include "pyopenvino/graph/axis_vector.hpp"
#include "pyopenvino/graph/coordinate.hpp"
#include "pyopenvino/graph/coordinate_diff.hpp"
#include "pyopenvino/graph/function.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/graph/node_factory.hpp"
#include "pyopenvino/graph/node_input.hpp"
#include "pyopenvino/graph/node_output.hpp"
#if defined(NGRAPH_ONNX_FRONTEND_ENABLE)
#    include "pyopenvino/graph/onnx_import/onnx_import.hpp"
#endif
#include "pyopenvino/graph/dimension.hpp"
#include "pyopenvino/graph/frontend/frontend.hpp"
#include "pyopenvino/graph/frontend/frontend_manager.hpp"
#include "pyopenvino/graph/frontend/inputmodel.hpp"
#include "pyopenvino/graph/frontend/place.hpp"
#include "pyopenvino/graph/ops/constant.hpp"
#include "pyopenvino/graph/ops/parameter.hpp"
#include "pyopenvino/graph/ops/result.hpp"
#include "pyopenvino/graph/ops/util/regmodule_graph_op_util.hpp"
#include "pyopenvino/graph/partial_shape.hpp"
#include "pyopenvino/graph/passes/regmodule_graph_passes.hpp"
#include "pyopenvino/graph/rt_map.hpp"
#include "pyopenvino/graph/shape.hpp"
#include "pyopenvino/graph/strides.hpp"
#include "pyopenvino/graph/types/regmodule_graph_types.hpp"
#include "pyopenvino/graph/util.hpp"
#include "pyopenvino/graph/variant.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyopenvino, m) {
    m.doc() = "Package pyopenvino that wraps OpenVino's namespace ov";
    regclass_graph_PyRTMap(m);
    regmodule_graph_types(m);
    regclass_graph_Dimension(m);  // Dimension must be registered before PartialShape
    regclass_graph_Shape(m);
    regclass_graph_PartialShape(m);
    regclass_graph_Node(m);
    regclass_graph_Place(m);
    regclass_graph_InitializationFailureFrontEnd(m);
    regclass_graph_GeneralFailureFrontEnd(m);
    regclass_graph_OpConversionFailureFrontEnd(m);
    regclass_graph_OpValidationFailureFrontEnd(m);
    regclass_graph_NotImplementedFailureFrontEnd(m);
    regclass_graph_FrontEndManager(m);
    regclass_graph_FrontEnd(m);
    regclass_graph_InputModel(m);
    regclass_graph_Input(m);
    regclass_graph_Output(m);
    regclass_graph_NodeFactory(m);
    regclass_graph_Strides(m);
    regclass_graph_CoordinateDiff(m);
    regclass_graph_AxisSet(m);
    regclass_graph_AxisVector(m);
    regclass_graph_Coordinate(m);
    py::module m_op = m.def_submodule("op", "Package ngraph.impl.op that wraps ngraph::op");  // TODO(!)
    regclass_graph_op_Constant(m_op);
    regclass_graph_op_Parameter(m_op);
    regclass_graph_op_Result(m_op);
#if defined(NGRAPH_ONNX_FRONTEND_ENABLE)
    regmodule_graph_onnx_import(m);
#endif
    regmodule_graph_op_util(m_op);
    regclass_graph_Function(m);
    regmodule_graph_passes(m);
    regmodule_graph_util(m);
    regclass_graph_Variant(m);
    regclass_graph_VariantWrapper<std::string>(m, std::string("String"));
    regclass_graph_VariantWrapper<int64_t>(m, std::string("Int"));
}
