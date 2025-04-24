// Copyright (C) 2018-2025 Intel Corporation
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
    return node.default_single_output_mapping({std::make_shared<ov::opset6::PRelu>(data, alpha)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
