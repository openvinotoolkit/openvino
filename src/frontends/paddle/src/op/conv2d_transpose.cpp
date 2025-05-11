// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv2d_utils.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs conv2d_transpose(const NodeContext& node) {
    return conv2d_base<opset6::GroupConvolutionBackpropData, opset6::ConvolutionBackpropData>(node);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
