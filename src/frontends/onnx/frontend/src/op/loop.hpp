// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief      Creates nGraph node representing ONNX loop operator.
///
/// \note       Details available here:
///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
///
/// \param[in]  node  The input ONNX node representing this operation.
///
/// \return     Vector of nodes containting resulting nGraph nodes.
///
OutputVector loop(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
