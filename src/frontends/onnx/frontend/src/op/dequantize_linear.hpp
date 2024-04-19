// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {

namespace set_1 {
ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node);

}  // namespace set_1

namespace set_13 {
namespace detail {
ov::OutputVector dequantize_linear(const ov::Output<ov::Node>& x,
                                   const ov::Output<ov::Node>& scale,
                                   const std::shared_ptr<ov::Node>& zero_point,
                                   int64_t axis,
                                   const Node& node);
}
ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node);
}  // namespace set_13
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
