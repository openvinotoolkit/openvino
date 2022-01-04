// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "onnx_import/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief Performs ONNX QLinearMatMul operation.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX quantizied
///         matrix multiplication operation.
OutputVector qlinear_matmul(const Node& node);
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ov
