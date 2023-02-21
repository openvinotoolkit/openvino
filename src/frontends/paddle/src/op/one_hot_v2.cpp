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
    Output<Node> depth_node;
    if (node.has_attribute("depth")) {
        depth_node = default_opset::Constant::create(element::i64, Shape{}, {node.get_attribute<int>("depth")});
    } else {
        depth_node = node.get_input("depth_tensor");
        if (depth_node.get_element_type() != element::i64) {
            depth_node = std::make_shared<default_opset::Convert>(depth_node, element::i64);
        }
    }
    auto on_value = default_opset::Constant::create(element::f32, Shape{}, {1});
    auto off_value = default_opset::Constant::create(element::f32, Shape{}, {0});
    auto one_hot = std::make_shared<ov::op::v1::OneHot>(data, depth_node, on_value, off_value, 1);
    return node.default_single_output_mapping({one_hot}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
