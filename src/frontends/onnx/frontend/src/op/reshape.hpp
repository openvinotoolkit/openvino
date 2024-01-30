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
///
/// \brief      Reshape the input tensor similar to numpy.reshape.
///
/// \param[in]  node  The ONNX node representing this operation.
///
/// \return     OV node representing this operation.
///
ov::OutputVector reshape(const ONNX_Node& node);

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
