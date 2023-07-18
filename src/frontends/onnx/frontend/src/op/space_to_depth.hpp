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
/// \brief      Permutes input tensor blocks of spatial data into depth.
///
/// \param[in]  node  The ONNX input node describing operation.
///
/// \return     OutputVector containing Tensor with shape:
///             [N, C * blocksize * blocksize, H / blocksize, W / blocksize]
OutputVector space_to_depth(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
