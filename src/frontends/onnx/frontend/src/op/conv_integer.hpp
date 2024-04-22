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
/// \brief Performs ONNX ConvInteger operation.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of quantized ONNX
///         convolution operation.
ov::OutputVector conv_integer(const ov::frontend::onnx::Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
