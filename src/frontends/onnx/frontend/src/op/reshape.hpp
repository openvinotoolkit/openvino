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
///
/// \brief      Reshape the input tensor similar to numpy.reshape.
///
/// \param[in]  node  The ONNX node representing this operation.
///
/// \return     OV node representing this operation.
///
ov::OutputVector reshape(const ov::frontend::onnx::Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
