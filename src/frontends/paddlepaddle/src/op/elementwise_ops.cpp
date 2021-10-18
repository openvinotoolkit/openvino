// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <node_context.hpp>

#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
template <typename T>
NamedOutputs elementwise_ops(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto y = node.get_ng_input("Y");

    auto axis = node.get_attribute<int>("axis");

    PDPD_OP_VALIDATION_CHECK(node, x.get_partial_shape().rank().is_static(), "elementwise_ops: X rank must be static!");
    PDPD_OP_VALIDATION_CHECK(node, y.get_partial_shape().rank().is_static(), "elementwise_ops: Y rank must be static!");
    int64_t x_rank = x.get_partial_shape().rank().get_length();
    int64_t y_rank = y.get_partial_shape().rank().get_length();

    if ((axis == -1) || (axis == x_rank - 1) || (x_rank == y_rank)) {
        return node.default_single_output_mapping({std::make_shared<T>(x, y)}, {"Out"});
    } else {
        std::vector<int64_t> indices;
        for (int64_t i = 0; i < axis; i++)
            indices.push_back(i);
        for (int64_t i = y_rank + axis; i < x_rank; i++)
            indices.push_back(i);

        auto indices_node =
            ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{indices.size()}, indices);
        auto y_node = std::make_shared<ngraph::opset6::Unsqueeze>(y, indices_node);
        return node.default_single_output_mapping({std::make_shared<T>(x, y_node)}, {"Out"});
    }
}

//
NamedOutputs elementwise_add(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Add>(node_context);
}

NamedOutputs elementwise_sub(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Subtract>(node_context);
}

NamedOutputs elementwise_mul(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Multiply>(node_context);
}

NamedOutputs elementwise_div(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Divide>(node_context);
}

NamedOutputs elementwise_min(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Minimum>(node_context);
}

NamedOutputs elementwise_max(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Maximum>(node_context);
}

NamedOutputs elementwise_pow(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Power>(node_context);
}

NamedOutputs elementwise_equal(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::Equal>(node_context);
}

NamedOutputs elementwise_greater_equal(const NodeContext& node_context) {
    return elementwise_ops<ov::opset6::GreaterEqual>(node_context);
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
