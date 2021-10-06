// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>
#include <pybind11/pybind11.h>

#include <ie_iinfer_request.hpp>
#include <ie_version.hpp>
#include <string>

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

#include "core/containers.hpp"
#include "core/ie_blob.hpp"
#include "core/ie_core.hpp"
#include "core/ie_data.hpp"
#include "core/ie_executable_network.hpp"
#include "core/ie_infer_queue.hpp"
#include "core/ie_infer_request.hpp"
#include "core/ie_input_info.hpp"
#include "core/ie_network.hpp"
#include "core/ie_parameter.hpp"
#include "core/ie_preprocess_info.hpp"
#include "core/ie_version.hpp"
#include "core/tensor_description.hpp"

namespace py = pybind11;

std::string get_version() {
    auto version = InferenceEngine::GetInferenceEngineVersion();
    std::string version_str = std::to_string(version->apiVersion.major) + ".";
    version_str += std::to_string(version->apiVersion.minor) + ".";
    version_str += version->buildNumber;
    return version_str;
}

PYBIND11_MODULE(pyopenvino, m) {
    m.doc() = "Package openvino.pyopenvino which wraps openvino C++ APIs";
    m.def("get_version", &get_version);

    py::enum_<InferenceEngine::StatusCode>(m, "StatusCode")
        .value("OK", InferenceEngine::StatusCode::OK)
        .value("GENERAL_ERROR", InferenceEngine::StatusCode::GENERAL_ERROR)
        .value("NOT_IMPLEMENTED", InferenceEngine::StatusCode::NOT_IMPLEMENTED)
        .value("NETWORK_NOT_LOADED", InferenceEngine::StatusCode::NETWORK_NOT_LOADED)
        .value("PARAMETER_MISMATCH", InferenceEngine::StatusCode::PARAMETER_MISMATCH)
        .value("NOT_FOUND", InferenceEngine::StatusCode::NOT_FOUND)
        .value("OUT_OF_BOUNDS", InferenceEngine::StatusCode::OUT_OF_BOUNDS)
        .value("UNEXPECTED", InferenceEngine::StatusCode::UNEXPECTED)
        .value("REQUEST_BUSY", InferenceEngine::StatusCode::REQUEST_BUSY)
        .value("RESULT_NOT_READY", InferenceEngine::StatusCode::RESULT_NOT_READY)
        .value("NOT_ALLOCATED", InferenceEngine::StatusCode::NOT_ALLOCATED)
        .value("INFER_NOT_STARTED", InferenceEngine::StatusCode::INFER_NOT_STARTED)
        .value("NETWORK_NOT_READ", InferenceEngine::StatusCode::NETWORK_NOT_READ)
        .export_values();

    py::enum_<InferenceEngine::IInferRequest::WaitMode>(m, "WaitMode")
        .value("RESULT_READY", InferenceEngine::IInferRequest::WaitMode::RESULT_READY)
        .value("STATUS_ONLY", InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY)
        .export_values();


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

    regclass_Core(m);
    regclass_IENetwork(m);

    regclass_Data(m);
    regclass_TensorDecription(m);

    // Registering template of Blob
    regclass_Blob(m);
    // Registering specific types of Blobs
    regclass_TBlob<float>(m, "Float32");
    regclass_TBlob<double>(m, "Float64");
    regclass_TBlob<int64_t>(m, "Int64");
    regclass_TBlob<uint64_t>(m, "Uint64");
    regclass_TBlob<int32_t>(m, "Int32");
    regclass_TBlob<uint32_t>(m, "Uint32");
    regclass_TBlob<int16_t>(m, "Int16");
    regclass_TBlob<uint16_t>(m, "Uint16");
    regclass_TBlob<int8_t>(m, "Int8");
    regclass_TBlob<uint8_t>(m, "Uint8");

    // Registering specific types of containers
    Containers::regclass_PyConstInputsDataMap(m);
    Containers::regclass_PyOutputsDataMap(m);
    Containers::regclass_PyResults(m);

    regclass_ExecutableNetwork(m);
    regclass_InferRequest(m);
    regclass_Version(m);
    regclass_Parameter(m);
    regclass_InputInfo(m);
    regclass_InferQueue(m);
    regclass_PreProcessInfo(m);
}
