// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

namespace ov {
namespace frontend {
namespace onnx {
namespace transform {

using ::ONNX_NAMESPACE::ModelProto;

static const std::vector<std::string> onnx_functions_to_expand = {"Bernoulli"};

/// \brief Replace nodes with expanded body of ONNX functions
///
/// Some ONNX operators are specified as functions, which can be expanded to
/// a subgraph or more primitive operations. This functions modifies the ONNX
/// model by replacing operations of types listed in onnx_functions_to_expand
/// with their expanded subgraphs.
///
/// \param model_proto Protobuf message with ONNX model to transform.
void expand_onnx_functions(ModelProto& model_proto);

}  // namespace transform
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
