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
/// \brief      Permutes input tensor data from depth into blocks of spatial data.
///
/// \note       Values from the depth dimension (assuming NCHW layout) are moved in
///             spatial blocks to the height and width dimensions.
///
/// \param[in]  node  The ONNX input node describing operation.
///
/// \return     ov::OutputVector containing Tensor with shape:
///             [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]
ov::OutputVector depth_to_space(const ov::frontend::onnx::Node& node);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
