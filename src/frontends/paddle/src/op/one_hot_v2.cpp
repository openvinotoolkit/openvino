// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs one_hot_v2(const NodeContext& node) {
    auto data = node.get_input("X");
    Output<Node> depth;
    if (node.has_input("depth_tensor")) {
        auto depth_value = node.get_input("depth_tensor");
        depth = std::make_shared<default_opset::Squeeze>(depth_value);
    } else {
        const auto depth_value = node.get_attribute<int>("depth");
        depth = default_opset::Constant::create(element::i32, Shape{}, {depth_value});
    }
    auto on_value = default_opset::Constant::create(element::f32, Shape{}, {1});
    auto off_value = default_opset::Constant::create(element::f32, Shape{}, {0});
    const auto indices_axis = -1;
    auto result = std::make_shared<default_opset::OneHot>(data, depth, on_value, off_value, indices_axis);
    return node.default_single_output_mapping({result}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
