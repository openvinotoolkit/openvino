// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ngraph/function.hpp"
#include "openvino/frontend/extension/holder.hpp"

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
/// \param      extensions An object containing a collection of frontend extensions to use during the import process
///
/// \return     An nGraph function that represents a single output from the created
/// graph.
std::shared_ptr<Function> import_onnx_model(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                            const std::string& model_path,
                                            ov::frontend::ExtensionHolder extensions = {});

/// \brief      Decode ONNX model to nGraph function with ONNXFrameworkNode(s)
///
/// \param      model_proto Reference to a GraphProto object.
/// \param      model_path  The path to the imported onnx model.
///                         It is required if the imported model uses data saved in external files.
/// \param      extensions An object containing a collection of frontend extensions to use during the import process
///
/// \return     A nGraph function with ONNXFrameworkNodes
std::shared_ptr<Function> decode_to_framework_nodes(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                                    const std::string& model_path,
                                                    ov::frontend::ExtensionHolder extensions = {});

/// \brief     Converts a nGraph function (onnx model decoded to function with ONNXFrameworkNode(s))
///            to a complete function with actual compute operations
///
/// \return    A nGraph function.
void convert_decoded_function(std::shared_ptr<Function> function);
}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
