// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
namespace detail {
ov::OutputVector conv(const ov::frontend::onnx::Node& node,
                      ov::Output<ov::Node> data,
                      ov::Output<ov::Node> filters,
                      ov::Output<ov::Node> bias);
}
/// \brief Performs ONNX Conv operation.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX convolution
///         operation.
ov::OutputVector conv(const ov::frontend::onnx::Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
