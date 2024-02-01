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
/// \brief      Creates OV node representing ONNX InstanceNormalization
///             operator.
///
/// \note       The resulting node represents following equation:
///             y = scale * (x - mean) / sqrt(variance + epsilon) + B
///             where mean and variance are computed per instance per channel.
///
/// \param[in]  node  The input ONNX node representing this operation.
///
/// \return     Vector of nodes containting resulting OV nodes.
///
ov::OutputVector instance_norm(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
