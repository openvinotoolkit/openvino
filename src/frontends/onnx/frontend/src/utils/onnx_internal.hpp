// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/frontend/extension/holder.hpp"
#include "utils/tensor_external_data.hpp"

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace ov {
namespace frontend {
namespace onnx {
namespace detail {

using ::ONNX_NAMESPACE::ModelProto;

/// \brief      Imports and converts an serialized ONNX model from a ModelProto
///             to an ov::Model representation.
///
/// \note       The function can be used only internally by OV components!
///             Passing ModelProto between componets which use different protobuf
///             library can cause segfaults. If stream parsing fails or the ONNX model
///             contains unsupported ops, the function throws an ov::Exception.
///
/// \param      model_proto Reference to a GraphProto object.
/// \param      model_path  The path to the imported onnx model.
///                         It is required if the imported model uses data saved in external files.
/// \param      enable_mmap Enable mapping files with external weights instead of reading.
/// \param      extensions An object containing a collection of frontend extensions to use during the import process
/// \return     An ov::Model that represents a single output from the created
/// graph.
std::shared_ptr<ov::Model> import_onnx_model(std::shared_ptr<ModelProto> model_proto,
                                             const std::string& model_path,
                                             detail::MappedMemoryHandles mmap_cache,
                                             ov::frontend::ExtensionHolder extensions = {});

/// \brief      Decode ONNX model to ov::Model with ONNXFrameworkNode(s)
///
/// \param      model_proto Reference to a GraphProto object.
/// \param      model_path  The path to the imported onnx model.
///                         It is required if the imported model uses data saved in external files.
/// \param      enable_mmap Enable mapping files with external weights instead of reading.
/// \param      extensions An object containing a collection of frontend extensions to use during the import process
/// \return     A ov::Model with ONNXFrameworkNodes
std::shared_ptr<ov::Model> decode_to_framework_nodes(std::shared_ptr<ModelProto> model_proto,
                                                     const std::string& model_path,
                                                     detail::MappedMemoryHandles mmap_cache,
                                                     ov::frontend::ExtensionHolder extensions = {});

/// \brief     Converts a ov::Model (onnx model decoded to function with ONNXFrameworkNode(s))
///            to a complete function with actual compute operations
void convert_decoded_model(std::shared_ptr<ov::Model> model);

}  // namespace detail
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
