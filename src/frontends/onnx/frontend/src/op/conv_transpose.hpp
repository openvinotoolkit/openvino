// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
/// \brief Performs ONNX Transposed Convolution operation.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX convolution
///         operation.
ov::OutputVector conv_transpose(const ov::frontend::onnx::Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
