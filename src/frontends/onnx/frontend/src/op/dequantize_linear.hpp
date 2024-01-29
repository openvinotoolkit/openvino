// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"
#include "openvino/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {

namespace set_1 {
ov::OutputVector dequantize_linear(const Node& node);

}  // namespace set_1

namespace set_13 {
namespace detail {
ov::OutputVector dequantize_linear(const ov::Output<ov::Node>& x,
                                   const ov::Output<ov::Node>& scale,
                                   const std::shared_ptr<ov::Node>& zero_point,
                                   int64_t axis,
                                   const Node& node);
}
ov::OutputVector dequantize_linear(const Node& node);
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
