// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_transpose(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    Output<Node> rank;
    std::tie(std::ignore, rank) = get_shape_rank(context, data, true);
    auto dim0_node = get_input_as_i32(context, 1);
    auto dim1_node = get_input_as_i32(context, 2);
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
    Output<Node> scatter = std::make_shared<v3::ScatterElementsUpdate>(range, indices, updates, axis_0);
    if (const auto scatter_const = ov::util::get_constant_from_source(scatter)) {
        scatter = context.mark_node(scatter_const);
    } else {
        context.mark_nodes(
            {start, step, range, axis_0, dim0_node_, dim1_node_, indices, updates, scatter.get_node_shared_ptr()});
    }

    return {context.mark_node(std::make_shared<v1::Transpose>(data, scatter))};
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
};

OutputVector translate_movedim(const NodeContext& context) {
    // aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
    // aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
    // based on https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp#L3816
    num_inputs_check(context, 3, 3);
    auto x = context.get_input(0);
    auto src_dims = get_input_as_i32(context, 1);
    auto dst_dims = get_input_as_i32(context, 2);
    Output<Node> rank;
    std::tie(std::ignore, rank) = get_shape_rank(context, context.get_input(0), true);
    src_dims = normalize_axis(context, src_dims, rank);
    dst_dims = normalize_axis(context, dst_dims, rank);
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, {}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, {}, {1}));
    auto range = context.mark_node(std::make_shared<v4::Range>(const_0, rank, const_1, element::i32));
    auto dims_1d_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    // operation accepts 0d and 1d source and destination, make them always 1d
    src_dims = context.mark_node(std::make_shared<v1::Reshape>(src_dims, dims_1d_shape, false));
    dst_dims = context.mark_node(std::make_shared<v1::Reshape>(dst_dims, dims_1d_shape, false));
    auto dims_shape = context.mark_node(std::make_shared<v3::ShapeOf>(src_dims, element::i32));
    auto minus_one_replaces = context.mark_node(std::make_shared<v1::Broadcast>(dims_1d_shape, dims_shape));
    // update position for the dim provided by user and mark used dims for source and destination as -1
    auto perm_dims = context.mark_node(std::make_shared<v3::ScatterElementsUpdate>(range, dst_dims, src_dims, const_0));
    auto src_perm_dims =
        context.mark_node(std::make_shared<v3::ScatterElementsUpdate>(range, src_dims, minus_one_replaces, const_0));
    auto dst_perm_dims =
        context.mark_node(std::make_shared<v3::ScatterElementsUpdate>(range, dst_dims, minus_one_replaces, const_0));
    // Remove the dims whose position we already know, the ones marked with -1 in previous step
    auto not_changed_src = context.mark_node(std::make_shared<v1::NotEqual>(src_perm_dims, dims_1d_shape));
    auto not_changed_dst = context.mark_node(std::make_shared<v1::NotEqual>(dst_perm_dims, dims_1d_shape));
    auto indices = context.mark_node(std::make_shared<v3::NonZero>(not_changed_dst, element::i32));
    auto updates = context.mark_node(std::make_shared<v3::NonZero>(not_changed_src, element::i32));
    // Update the position of the remaining dimensions. indices now contains the original position
    // updates contains the new position it will shifted to after considering the user inputs.
    indices = context.mark_node(std::make_shared<v1::Reshape>(indices, dims_1d_shape, false));
    updates = context.mark_node(std::make_shared<v1::Reshape>(updates, dims_1d_shape, false));
    auto scatter = std::make_shared<v3::ScatterElementsUpdate>(perm_dims, indices, updates, const_0);
    return {context.mark_node(std::make_shared<v1::Transpose>(x, scatter))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
