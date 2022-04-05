// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
namespace detail {
OutputVector conv(const Node& node, Output<ngraph::Node> data, Output<ngraph::Node> filters, Output<ngraph::Node> bias);
}
/// \brief Performs ONNX Conv operation.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing Ngraph nodes producing output of ONNX convolution
///         operation.
OutputVector conv(const Node& node);

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
