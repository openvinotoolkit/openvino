// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ngraph/function.hpp"
#include "openvino/frontend/extension/progress_reporter_extension.hpp"
#include "openvino/frontend/extension/telemetry.hpp"

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace ngraph {
namespace onnx_import {
namespace detail {
/// \brief      Imports and converts an serialized ONNX model from a ModelProto
///             to an nGraph Function representation.
///
/// \note       The function can be used only internally by OV components!
///             Passing ModelProto between componets which use different protobuf
///             library can cause segfaults. If stream parsing fails or the ONNX model
///             contains unsupported ops, the function throws an ngraph_error exception.
///
/// \param      model_proto Reference to a GraphProto object.
/// \param      model_path  The path to the imported onnx model.
///                         It is required if the imported model uses data saved in external files.
/// \param      telemetry An extension used to gather information about models being imported
/// \param      progress_reporter An extension used to notify about the overall progress of the model importing process
///
/// \return     An nGraph function that represents a single output from the created
/// graph.
std::shared_ptr<Function> import_onnx_model(
    std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
    const std::string& model_path,
    const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry = {},
    const std::shared_ptr<ov::frontend::ProgressReporterExtension>& progress_reporter = {});

/// \brief      Decode ONNX model to nGraph function with ONNXFrameworkNode(s)
///
/// \param      model_proto Reference to a GraphProto object.
/// \param      model_path  The path to the imported onnx model.
///                         It is required if the imported model uses data saved in external files.
/// \param      telemetry An extension used to gather information about models being imported
/// \param      progress_reporter An extension used to notify about the overall progress of the model importing process
///
/// \return     A nGraph function with ONNXFrameworkNodes
std::shared_ptr<Function> decode_to_framework_nodes(
    std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
    const std::string& model_path,
    const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry = {},
    const std::shared_ptr<ov::frontend::ProgressReporterExtension>& progress_reporter = {});

/// \brief     Converts a nGraph function (onnx model decoded to function with ONNXFrameworkNode(s))
///            to a complete function with actual compute operations
///
/// \return    A nGraph function.
void convert_decoded_function(std::shared_ptr<Function> function);
}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
