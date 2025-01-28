// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise_ops.hpp"

#include "op_utils.hpp"

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
    auto x = node_context.get_input("X");
    if (x.get_element_type() == ov::element::boolean)
        return elementwise_ops<default_opset::LogicalAnd>(node_context);
    else
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

NamedOutputs equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::Equal>(node_context);
}

NamedOutputs greater_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::GreaterEqual>(node_context);
}

NamedOutputs not_equal(const NodeContext& node_context) {
    return elementwise_ops<default_opset::NotEqual>(node_context);
}

NamedOutputs less_equal(const NodeContext& node) {
    return elementwise_ops<default_opset::LessEqual>(node);
}

NamedOutputs elementwise_floordiv(const NodeContext& node_context) {
    auto x = node_context.get_input("X");
    auto y = node_context.get_input("Y");
    auto axis = -1;
    if (node_context.has_attribute("axis")) {
        axis = node_context.get_attribute<int>("axis");
    }

    int64_t pd_version = node_context.get_version();

    bool python_div = false;
    if (pd_version >= 2005000 || pd_version == 0) {
        python_div = true;
    }
    x = get_tensor_safe(x);
    y = get_tensor_safe(y);
    return node_context.default_single_output_mapping(
        {std::make_shared<default_opset::Divide>(x,
                                                 y,
                                                 python_div,
                                                 ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, axis))},
        {"Out"});
}

NamedOutputs elementwise_mod(const NodeContext& node_context) {
    return elementwise_ops<default_opset::FloorMod>(node_context);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
