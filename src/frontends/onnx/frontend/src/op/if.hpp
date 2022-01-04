// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_vector.hpp"
#include "onnx_import/core/node.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief Convert ONNX If operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX If
///        operation.
OutputVector if_op(const Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ov
