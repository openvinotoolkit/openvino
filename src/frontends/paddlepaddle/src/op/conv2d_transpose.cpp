// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "conv2d_utils.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddlepaddle {
namespace op {
NamedOutputs conv2d_transpose(const NodeContext& node) {
    return conv2d_base<opset6::GroupConvolutionBackpropData, opset6::ConvolutionBackpropData>(node);
}

}  // namespace op
}  // namespace paddlepaddle
}  // namespace frontend
}  // namespace ov
