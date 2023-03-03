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
    auto x = node.get_input("X");
    auto depth = node.get_attribute<int>("depth");
    auto x_type = x.get_element_type();
    auto depth_node = default_opset::Constant::create(x_type, Shape{}, {depth});
    auto on_value_node = default_opset::Constant::create(element::f32, Shape{}, {1.0});
    auto off_value_node = default_opset::Constant::create(element::f32, Shape{}, {0.0});
    auto axis = node.get_attribute<int64_t>("axis", -1);
    std::shared_ptr<Node> out;
    out = std::make_shared<default_opset::OneHot>(x, depth_node, on_value_node, off_value_node, axis);
    return node.default_single_output_mapping({out}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
