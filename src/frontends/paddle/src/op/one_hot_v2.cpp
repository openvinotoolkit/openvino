// Copyright (C) 2018-2023 Intel Corporation
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
    auto depth_node;
    if (node.has_attribute("depth")) {
        depth_node = node.get_attribute<int64_t>("depth");
    } else {
        depth_node = node.get_input("depth_tensor");
    }
    auto depth = std::make_shared<default_opset::Convert>(depth_node, element::i64);
    // paddle::one_hot_v2 has only one input, so we need to create a constant for on_value

    auto on_value = default_opset::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = default_opset::Constant::create(element::f32, Shape{}, {0.0f});
    auto one_hot = std::make_shared<default_opset::OneHot>(data, depth, on_value, off_value);
    return node.default_single_output_mapping({one_hot}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
