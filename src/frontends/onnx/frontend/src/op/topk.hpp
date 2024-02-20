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
/// \brief Performs ONNX TopK operation.
///
/// \param node The ONNX node object representing this operation.
/// \return The vector containing OV nodes producing output of ONNX TopK
///         operation (both values and indices).
ov::OutputVector topk(const ov::frontend::onnx::Node& node);
}  // namespace set_1

/// \brief Performs TopK operation from ONNX version 1.5
///
/// \details ONNX op set 10 added support for K as a dynamic input, not a static
/// attribute.
namespace set_10 {
ov::OutputVector topk(const ov::frontend::onnx::Node& node);
}

/// \brief Performs TopK operation from ONNX version 1.6
///
/// \details ONNX op set 11 added support for `largest` and `sorted` attributes.
namespace set_11 {
ov::OutputVector topk(const ov::frontend::onnx::Node& node);
}

}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
