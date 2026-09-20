// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs leaky_relu(const NodeContext& node) {
    auto data = node.get_input("X");
    auto alpha = ov::opset6::Constant::create(ov::element::f32, {1}, {node.get_attribute<float>("alpha")});
    // PRelu expects data and slope to have the same element type, so cast the alpha
    // (always read as f32) to the input type the same way dropout does for its scale.
    auto alpha_like = std::make_shared<ov::opset6::ConvertLike>(alpha, data);
    return node.default_single_output_mapping({std::make_shared<ov::opset6::PRelu>(data, alpha_like)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
