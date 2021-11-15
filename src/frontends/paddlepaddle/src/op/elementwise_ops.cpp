// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <ngraph/opsets/opset6.hpp>
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

    auto axis_value = node.get_attribute<int>("axis");
    if (axis_value == -1)
        return node.default_single_output_mapping({std::make_shared<T>(x, y)}, {"Out"});
    else {
        auto axis = ngraph::opset6::Constant::create(ngraph::element::i64, ov::Shape{}, {axis_value});

        auto x_shape = std::make_shared<ngraph::opset6::ShapeOf>(x);
        auto x_rank = std::make_shared<ngraph::opset6::ShapeOf>(x_shape);
        auto y_shape = std::make_shared<ngraph::opset6::ShapeOf>(y);
        auto y_rank = std::make_shared<ngraph::opset6::ShapeOf>(y_shape);

        auto const_0 = ngraph::opset6::Constant::create(ngraph::element::i64, ov::Shape{}, {0});
        auto const_1 = ngraph::opset6::Constant::create(ngraph::element::i64, ov::Shape{}, {1});

        /*  Use if op to handle when to broadcast y or not:
         *  if ((axis == x_rank - 1) || (x_rank == y_rank))
         *      not broadcast y;
         *  else
         *      broadcast y;
         */
        auto x_rank_sub_1 = std::make_shared<ngraph::opset6::Subtract>(x_rank, const_1);
        auto axis_is_last = std::make_shared<ngraph::opset6::Equal>(axis, x_rank_sub_1);
        auto rank_is_equal = std::make_shared<ngraph::opset6::Equal>(x_rank, y_rank);
        auto not_broadcast = std::make_shared<ngraph::opset6::LogicalOr>(axis_is_last, rank_is_equal);
        auto if_op = std::make_shared<ngraph::op::v8::If>(not_broadcast);
        auto element_type = y.get_element_type();
        // not broadcast
        auto then_x = std::make_shared<ngraph::op::Parameter>(element_type, PartialShape::dynamic());
        auto then_y = std::make_shared<ngraph::op::Parameter>(element_type, PartialShape::dynamic());
        auto then_res = std::make_shared<ngraph::op::Result>(then_y);
        auto then_body =
            std::make_shared<ngraph::Function>(ngraph::OutputVector{then_res}, ngraph::ParameterVector{then_x, then_y});
        // broadcast
        auto else_x = std::make_shared<ngraph::op::Parameter>(element_type, PartialShape::dynamic());
        auto else_y = std::make_shared<ngraph::op::Parameter>(element_type, PartialShape::dynamic());
        auto x_shape_else = std::make_shared<ngraph::opset6::ShapeOf>(else_x);
        auto x_rank_else = std::make_shared<ngraph::opset6::ShapeOf>(x_shape_else);
        auto y_shape_else = std::make_shared<ngraph::opset6::ShapeOf>(else_y);
        auto y_rank_else = std::make_shared<ngraph::opset6::ShapeOf>(y_shape_else);

        auto range0 = std::make_shared<ngraph::opset6::Range>(const_0, axis, const_1, ngraph::element::i64);
        auto y_rank_add_axis = std::make_shared<ngraph::opset6::Add>(y_rank_else, axis);
        auto y_rank_add_axis_scalar = std::make_shared<ngraph::opset6::Squeeze>(y_rank_add_axis);
        auto x_rank_scalar = std::make_shared<ngraph::opset6::Squeeze>(x_rank_else);
        auto range1 = std::make_shared<ngraph::opset6::Range>(y_rank_add_axis_scalar,
                                                              x_rank_scalar,
                                                              const_1,
                                                              ngraph::element::i64);
        auto indices = std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{range0, range1}, 0);
        auto y_broadcast = std::make_shared<ngraph::opset6::Unsqueeze>(else_y, indices);
        auto else_res = std::make_shared<ngraph::op::Result>(y_broadcast);
        auto else_body =
            std::make_shared<ngraph::Function>(ngraph::OutputVector{else_res}, ngraph::ParameterVector{else_x, else_y});
        // set input/output/body of if op
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(x, then_x, else_x);
        if_op->set_input(y, then_y, else_y);
        auto res_y = if_op->set_output(then_res, else_res);

        return node.default_single_output_mapping({std::make_shared<T>(x, res_y)}, {"Out"});
    }
}

//
NamedOutputs elementwise_add(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Add>(node_context);
}

NamedOutputs elementwise_sub(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Subtract>(node_context);
}

NamedOutputs elementwise_mul(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Multiply>(node_context);
}

NamedOutputs elementwise_div(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Divide>(node_context);
}

NamedOutputs elementwise_min(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Minimum>(node_context);
}

NamedOutputs elementwise_max(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Maximum>(node_context);
}

NamedOutputs elementwise_pow(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Power>(node_context);
}

NamedOutputs elementwise_equal(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::Equal>(node_context);
}

NamedOutputs elementwise_greater_equal(const NodeContext& node_context) {
    return elementwise_ops<ngraph::opset6::GreaterEqual>(node_context);
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
