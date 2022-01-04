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
/// \brief Convert ONNX ArgMin operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing an OV node which produces the output
///         of an ONNX ArgMin operation.
OutputVector argmin(const Node& node);

}  // namespace set_1

namespace set_12 {
/// \brief Convert ONNX ArgMin operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing an OV node which produces the output
///         of an ONNX ArgMax operation.
OutputVector argmin(const Node& node);

}  // namespace set_12

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
