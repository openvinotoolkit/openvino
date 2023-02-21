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
    auto loss = node.get_input("x");
    auto lambda  = node.get_attribute<float>("lambda", 0.5);
    //initialize
    auto pos_lam = default_opset::Constant::create(element::f32, {1}, {lambda});
    auto neg_lam = default_opset::Constant::create(element::f32, {1}, {-lambda});
    auto zero = default_opset::Constant::create(element::f32, {1}, 0);
    //comparison
    auto pos_mask = std::make_shared<default_opset::Greater>(loss, pos_lam);
    auto neg_mask = std::make_shared<default_opset::Greater>(neg_lam, loss);
    //select
    auto positive = std::make_shared<default_opset::Subtract>(loss, pos_lam);
    auto negative = std::make_shared<default_opset::Add>(loss, neg_lam);
    auto mid_result = std::make_shared<default_opset::Select>(pos_mask, positive, zero);
    auto result = std::make_shared<default_opset::Select>(neg_mask, negative, mid_result);
    return node.default_single_output_mapping({result}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
