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
///
/// \brief      Reshape the input tensor similar to numpy.reshape.
///
/// \param[in]  node  The ONNX node representing this operation.
///
/// \return     OV node representing this operation.
///
OutputVector reshape(const Node& node);

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
