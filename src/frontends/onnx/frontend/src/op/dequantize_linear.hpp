// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "onnx_import/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace detail {
Output<ov::Node> get_zero_point(const OutputVector& inputs);
}

namespace set_1 {
OutputVector dequantize_linear(const Node& node);

}  // namespace set_1

namespace set_13 {
namespace detail {
OutputVector dequantize_linear(Output<ov::Node> x,
                               Output<ov::Node> scale,
                               Output<ov::Node> zero_point,
                               int64_t axis,
                               Node node);
}
OutputVector dequantize_linear(const Node& node);
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
