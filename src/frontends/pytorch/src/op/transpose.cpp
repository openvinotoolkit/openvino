// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_transpose(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    Output<Node> rank;
    std::tie(std::ignore, rank) = get_shape_rank(context, context.get_input(0), true);
    auto dim0_node = context.get_input(1);
    auto dim1_node = context.get_input(2);
    dim0_node = normalize_axis(context, dim0_node, rank);
    dim1_node = normalize_axis(context, dim1_node, rank);
    auto start = v0::Constant::create(element::i32, {}, {0});
    auto step = v0::Constant::create(element::i32, {}, {1});
    auto range = std::make_shared<v4::Range>(start, rank, step, element::i32);

    auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto dim0_node_ = std::make_shared<v0::Unsqueeze>(dim0_node, axis_0);
    auto dim1_node_ = std::make_shared<v0::Unsqueeze>(dim1_node, axis_0);
    auto indices = std::make_shared<v0::Concat>(OutputVector{dim0_node_, dim1_node_}, 0);
    auto updates = std::make_shared<v0::Concat>(OutputVector{dim1_node_, dim0_node_}, 0);
    auto scatter = std::make_shared<v3::ScatterElementsUpdate>(range, indices, updates, axis_0);
    context.mark_nodes({start, step, range, axis_0, dim0_node_, dim1_node_, indices, updates, scatter});

    return {context.mark_node(std::make_shared<v1::Transpose>(context.get_input(0), scatter))};
};

OutputVector translate_t(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    if (input.get_partial_shape().rank().is_static()) {
        if (input.get_partial_shape().rank().get_length() < 2) {
            return {input};
        }
        auto dims = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
        return {context.mark_node(std::make_shared<v1::Transpose>(input, dims))};
    } else {
        // If rank is not known we create If operation
        Output<Node> rank;
        std::tie(std::ignore, rank) = get_shape_rank(context, input, true);
        auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
        auto cond = context.mark_node(std::make_shared<v1::Equal>(rank, const_2));

        // then body
        auto param_then = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
        auto dims = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
        auto transpose = context.mark_node(std::make_shared<v1::Transpose>(param_then, dims));
        auto result_then = std::make_shared<v0::Result>(transpose);
        auto then_body = std::make_shared<Model>(ResultVector{result_then}, ParameterVector{param_then});

        // else body
        auto param_else = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
        auto result_else = std::make_shared<v0::Result>(param_else);
        auto else_body = std::make_shared<Model>(ResultVector{result_else}, ParameterVector{param_else});

        // If op creation
        auto if_node = std::make_shared<v8::If>(cond);
        context.mark_node(if_node);
        if_node->set_then_body(then_body);
        if_node->set_else_body(else_body);
        if_node->set_input(input, param_then, param_else);
        return {if_node->set_output(result_then, result_else)};
    }
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
