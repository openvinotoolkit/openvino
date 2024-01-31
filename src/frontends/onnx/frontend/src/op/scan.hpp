// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief      Creates OpenVino node representing ONNX Scan operator.
///
/// \note       Details available here:
///             https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scan
///
/// \param[in]  node  The input ONNX node representing this operation.
///
/// \return     ov::OutputVector of resulting OpenVino nodes.
///
ov::OutputVector scan(const Node& node);
}  // namespace set_1

namespace set_9 {
ov::OutputVector scan(const Node& node);
}  // namespace set_9

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
