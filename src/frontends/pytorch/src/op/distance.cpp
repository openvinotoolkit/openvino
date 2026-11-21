// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {
Output<Node> pairwise_distance(const NodeContext& context,
                               Output<Node> x,
                               Output<Node> y,
                               Output<Node> p,
                               Output<Node> eps,
                               bool keepdim) {
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto p_plus_eps = context.mark_node(std::make_shared<v1::Add>(p, eps));
    auto inv_p = context.mark_node(std::make_shared<v1::Divide>(one, p_plus_eps));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    align_eltwise_input_types(context, x, y, is_python_scalar_input(context, 0), is_python_scalar_input(context, 1));
    auto x_y_diff = context.mark_node(std::make_shared<v1::Subtract>(x, y));
    auto x_y_diff_in_p_power = context.mark_node(std::make_shared<v1::Power>(x_y_diff, p));
    auto summation = context.mark_node(std::make_shared<v1::ReduceSum>(x_y_diff_in_p_power, minus_one, keepdim));
    auto summation_in_inv_p = context.mark_node(std::make_shared<v1::Power>(summation, inv_p));
    return summation_in_inv_p;
}
};  // namespace

OutputVector translate_cdist(const NodeContext& context) {
    // aten::cdist(Tensor x1, Tensor x2, float p=2., int? compute_mode=None) -> Tensor
    // compute_mode can be ignored as we will always use matrix multiplication for euclidean distance computation
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    Output<Node> p = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0}));
    if (!context.input_is_none(2)) {
        p = context.get_input(2);
        p = context.mark_node(std::make_shared<v0::Convert>(p, element::f32));
    }
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));
    auto input_rank = context.mark_node(std::make_shared<v3::ShapeOf>(input_shape, element::i32));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
    auto x_unsqueeze_ax = context.mark_node(std::make_shared<v1::Subtract>(input_rank, one));
    auto y_unsqueeze_ax = context.mark_node(std::make_shared<v1::Subtract>(input_rank, two));
    auto x_unsqueeze = context.mark_node(std::make_shared<v0::Unsqueeze>(x, x_unsqueeze_ax));
    auto y_unsqueeze = context.mark_node(std::make_shared<v0::Unsqueeze>(y, y_unsqueeze_ax));
    auto eps = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1e-06}));
    auto result = pairwise_distance(context, x_unsqueeze, y_unsqueeze, p, eps, false);
    return {result};
};

OutputVector translate_pairwise_distance(const NodeContext& context) {
    // aten::pairwise_distance(Tensor x1, Tensor x2, float p=2., float eps=9.9999999999999995e-07, bool keepdim=False)
    num_inputs_check(context, 2, 5);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    Output<Node> p = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0}));
    Output<Node> eps = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1e-06}));
    bool keepdims = false;
    if (!context.input_is_none(2)) {
        p = context.get_input(2);
        p = context.mark_node(std::make_shared<v0::Convert>(p, element::f32));
    }
    if (!context.input_is_none(3)) {
        eps = context.get_input(3);
        eps = context.mark_node(std::make_shared<v0::Convert>(eps, element::f32));
    }
    if (!context.input_is_none(4)) {
        keepdims = context.const_input<bool>(4);
    }
    return {pairwise_distance(context, x, y, p, eps, keepdims)};
}

OutputVector translate_cosine_similarity(const NodeContext& context) {
    // aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
    num_inputs_check(context, 2, 4);
    auto x1 = context.get_input(0);
    auto x2 = context.get_input(1);
    align_eltwise_input_types(context, x1, x2, is_python_scalar_input(context, 0), is_python_scalar_input(context, 1));
    Output<Node> dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    if (!context.input_is_none(2)) {
        dim = get_input_as_i32(context, 2);
    }
    Output<Node> eps = context.mark_node(v0::Constant::create(element::f64, Shape{}, {1e-08}));
    if (!context.input_is_none(3)) {
        eps = context.get_input(3);
    }
    eps = context.mark_node(std::make_shared<v1::ConvertLike>(eps, x1));
    // PyTorch broadcasts the inputs before taking the norms, so a size-1 input along dim is
    // repeated first. Each norm is then clamped to eps separately, not their product.
    auto x1_shape = context.mark_node(std::make_shared<v3::ShapeOf>(x1, element::i32));
    auto x2_shape = context.mark_node(std::make_shared<v3::ShapeOf>(x2, element::i32));
    x1 = context.mark_node(std::make_shared<v3::Broadcast>(x1, x2_shape, BroadcastType::BIDIRECTIONAL));
    x2 = context.mark_node(std::make_shared<v3::Broadcast>(x2, x1_shape, BroadcastType::BIDIRECTIONAL));
    auto x1_norm = context.mark_node(std::make_shared<v4::ReduceL2>(x1, dim, true));
    auto x2_norm = context.mark_node(std::make_shared<v4::ReduceL2>(x2, dim, true));
    x1_norm = context.mark_node(std::make_shared<v1::Maximum>(x1_norm, eps));
    x2_norm = context.mark_node(std::make_shared<v1::Maximum>(x2_norm, eps));
    auto x1_normalized = context.mark_node(std::make_shared<v1::Divide>(x1, x1_norm));
    auto x2_normalized = context.mark_node(std::make_shared<v1::Divide>(x2, x2_norm));
    auto product = context.mark_node(std::make_shared<v1::Multiply>(x1_normalized, x2_normalized));
    return {context.mark_node(std::make_shared<v1::ReduceSum>(product, dim, false))};
}

}  // namespace ov::frontend::pytorch::op
