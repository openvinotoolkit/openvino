// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"
#include "openvino/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
namespace detail {
OutputVector conv(const Node& node, Output<ov::Node> data, Output<ov::Node> filters, Output<ov::Node> bias);
}
/// \brief Performs ONNX Conv operation.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX convolution
///         operation.
OutputVector conv(const Node& node);

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
