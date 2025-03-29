// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv2d_utils.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs conv2d(const NodeContext& node) {
    return conv2d_base<opset6::GroupConvolution, opset6::Convolution>(node);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
