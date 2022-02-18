// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
Output<ngraph::Node> get_zero_point(const OutputVector& inputs);
}

namespace set_1 {
OutputVector dequantize_linear(const Node& node);

}  // namespace set_1

namespace set_13 {
namespace detail {
OutputVector dequantize_linear(Output<ngraph::Node> x,
                               Output<ngraph::Node> scale,
                               Output<ngraph::Node> zero_point,
                               int64_t axis,
                               Node node);
}
OutputVector dequantize_linear(const Node& node);
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
