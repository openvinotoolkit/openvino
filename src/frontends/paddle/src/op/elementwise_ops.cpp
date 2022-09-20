// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

//
NamedOutputs elementwise_add(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Add>(node_context);
}

NamedOutputs elementwise_sub(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Subtract>(node_context);
}

NamedOutputs elementwise_mul(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Multiply>(node_context);
}

NamedOutputs elementwise_div(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Divide>(node_context);
}

NamedOutputs elementwise_min(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Minimum>(node_context);
}

NamedOutputs elementwise_max(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Maximum>(node_context);
}

NamedOutputs elementwise_pow(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Power>(node_context);
}

NamedOutputs elementwise_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Equal>(node_context);
}

NamedOutputs elementwise_greater_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::GreaterEqual>(node_context);
}

NamedOutputs elementwise_floordiv(const NodeContext& node_context) {
    auto x = node_context.get_input("X");
    auto y = node_context.get_input("Y");

    const auto axis = node_context.get_attribute<int>("axis", -1);
    auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, axis);

    return node_context.default_single_output_mapping({std::make_shared<default_opset::Divide>(x, y, autob)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
