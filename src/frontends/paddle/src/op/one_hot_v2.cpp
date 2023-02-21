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
    auto data = node.get_input("x"); 
    auto classes = node.get_attribute<int>("num_classes");
    auto depth = default_opset::Constant::create(element::i32, {}, {classes});
    auto on_value = default_opset::Constant::create(element::i32, {}, {0});
    auto off_value = default_opset::Constant::create(element::i32, {}, {1});
    const auto indices_axis = 1;
    auto one_hot = std::make_shared<default_opset::OneHot>(data, depth, on_value, off_value, indices_axis);
    return node.default_single_output_mapping({one_hot}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

