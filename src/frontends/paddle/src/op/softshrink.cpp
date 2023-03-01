// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs softshrink(const NodeContext& node) {
    auto x = node.get_input("X");
    auto threshold = node.get_attribute<float>("lambda");
    PADDLE_OP_CHECK(node, threshold >= 0.0f, "threshold must >= 0.");
    auto threshold_node = default_opset::Constant::create(ov::element::f32, {1}, {threshold});
    auto neg_threshold_node = default_opset::Constant::create(ov::element::f32, {1}, {-threshold});
    auto greater_threshold_node = std::make_shared<default_opset::Greater>(x, threshold_node);
    auto less_neg_threshold_node = std::make_shared<default_opset::Less>(x, neg_threshold_node);

    auto dtype = node.get_attribute<ov::element::Type>("dtype", element::f32);
    auto zero_value_node = default_opset::Constant::create(dtype, {1}, {0.0});
    auto shape_node = std::make_shared<default_opset::ShapeOf>(x);
    auto out_zero_node = std::make_shared<default_opset::Broadcast>(zero_value_node, shape_node);

    auto x_sub_threshold_node = std::make_shared<default_opset::Subtract>(x, threshold_node);
    auto x_add_threshold_node = std::make_shared<default_opset::Add>(x, threshold_node);

    std::shared_ptr<Node> out;
    auto out_greater_threshold_node = std::make_shared<default_opset::Select>(greater_threshold_node, x_sub_threshold_node, out_zero_node);
    out = std::make_shared<default_opset::Select>(less_neg_threshold_node, x_add_threshold_node, out_greater_threshold_node);

    return node.default_single_output_mapping({out}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
