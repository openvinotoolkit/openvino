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
/// \brief      Permutes input tensor blocks of spatial data into depth.
///
/// \param[in]  node  The ONNX input node describing operation.
///
/// \return     ov::OutputVector containing Tensor with shape:
///             [N, C * blocksize * blocksize, H / blocksize, W / blocksize]
ov::OutputVector space_to_depth(const ov::frontend::onnx::Node& node);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
