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
/// \brief Convert ONNX ArgMin operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing an OV node which produces the output
///         of an ONNX ArgMin operation.
ov::OutputVector argmin(const ov::frontend::onnx::Node& node);

}  // namespace set_1

namespace set_12 {
/// \brief Convert ONNX ArgMin operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing an OV node which produces the output
///         of an ONNX ArgMax operation.
ov::OutputVector argmin(const ov::frontend::onnx::Node& node);

}  // namespace set_12
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
