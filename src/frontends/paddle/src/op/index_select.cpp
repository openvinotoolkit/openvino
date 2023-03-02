// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs index_select(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto index = node.get_input("Index");
    Output<Node> axis;
    const auto axis_value = node.get_attribute<int>("dim", 0);
    axis = default_opset::Constant::create(element::i32, Shape{}, {axis_value});
    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Gather>(x, index, axis)},
        {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
