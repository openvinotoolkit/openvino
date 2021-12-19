// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv2d_utils.hpp"
#include "openvino/opsets/opset6.hpp"
#include "paddlepaddle_frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs conv2d_transpose(const NodeContext& node) {
    return conv2d_base<opset6::GroupConvolutionBackpropData, opset6::ConvolutionBackpropData>(node);
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
