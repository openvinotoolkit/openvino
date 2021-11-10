// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
template <typename T>
NamedOutputs logical_ops(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto y = node.get_ng_input("Y");
    return node.default_single_output_mapping({std::make_shared<T>(x, y)}, {"Out"});
}

NamedOutputs logical_and(const NodeContext& node_context) {
    return logical_ops<default_opset::LogicalAnd>(node_context);
}

NamedOutputs logical_or(const NodeContext& node_context) {
    return logical_ops<default_opset::LogicalOr>(node_context);
}

NamedOutputs logical_xor(const NodeContext& node_context) {
    return logical_ops<default_opset::LogicalXor>(node_context);
}

NamedOutputs logical_not(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    return node.default_single_output_mapping({std::make_shared<default_opset::LogicalNot>(data)}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
