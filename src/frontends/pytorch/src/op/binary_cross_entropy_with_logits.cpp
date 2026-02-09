// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// Negate a tensor using Multiply by -1 (same pattern as translate_neg).
// Avoids v0::Negative which can fail in ConvertNegative pass when element type is dynamic.
static Output<Node> negate(const NodeContext& context, const Output<Node>& x) {
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto cast = context.mark_node(std::make_shared<v1::ConvertLike>(const_neg_1, x));
    return context.mark_node(std::make_shared<v1::Multiply>(x, cast));
}

OutputVector translate_binary_cross_entropy_with_logits(const NodeContext& context) {
    // aten::binary_cross_entropy_with_logits(Tensor self, Tensor target,
    //     Tensor? weight=None, Tensor? pos_weight=None, int reduction=1) -> Tensor
    num_inputs_check(context, 2, 5);

    auto input = get_input_with_floating_type(context, 0);
    auto target = context.get_input(1);
    target = context.mark_node(std::make_shared<v1::ConvertLike>(target, input));

    // Numerically stable log_sigmoid(x) = -softplus(-x)
    auto neg_input = negate(context, input);
    auto softplus_neg = context.mark_node(std::make_shared<v4::SoftPlus>(neg_input));
    Output<Node> log_sigmoid_input = negate(context, softplus_neg);

    // Optional pos_weight (index 3): log_weight = (pos_weight - 1) * target + 1
    if (context.get_input_size() > 3 && !context.input_is_none(3)) {
        auto pos_weight = context.get_input(3);
        pos_weight = context.mark_node(std::make_shared<v1::ConvertLike>(pos_weight, input));

        auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0f}));
        one = context.mark_node(std::make_shared<v1::ConvertLike>(one, input));

        auto pw_minus_one = context.mark_node(std::make_shared<v1::Subtract>(pos_weight, one));
        auto pw_times_target = context.mark_node(std::make_shared<v1::Multiply>(pw_minus_one, target));
        auto log_weight = context.mark_node(std::make_shared<v1::Add>(pw_times_target, one));

        log_sigmoid_input = context.mark_node(std::make_shared<v1::Multiply>(log_sigmoid_input, log_weight));
    }

    // loss = (1 - target) * input - log_sigmoid_input
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0f}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, input));
    auto one_minus_target = context.mark_node(std::make_shared<v1::Subtract>(one, target));
    auto term1 = context.mark_node(std::make_shared<v1::Multiply>(one_minus_target, input));
    Output<Node> loss = context.mark_node(std::make_shared<v1::Subtract>(term1, log_sigmoid_input));

    // Optional weight (index 2): loss *= weight
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        auto weight = context.get_input(2);
        weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, input));
        loss = context.mark_node(std::make_shared<v1::Multiply>(loss, weight));
    }

    // reduction: 0=None, 1=Mean (default), 2=Sum
    int64_t reduction = 1;
    if (context.get_input_size() > 4 && !context.input_is_none(4)) {
        reduction = context.const_input<int64_t>(4);
    }

    if (reduction == 0) {
        return {loss};
    }

    auto axes = get_axes_range(context, 0);

    if (reduction == 1) {
        loss = context.mark_node(std::make_shared<v1::ReduceMean>(loss, axes, false));
        return {loss};
    }

    if (reduction == 2) {
        loss = context.mark_node(std::make_shared<v1::ReduceSum>(loss, axes, false));
        return {loss};
    }

    PYTORCH_OP_CONVERSION_CHECK(false,
                                "aten::binary_cross_entropy_with_logits: unsupported reduction value: ",
                                reduction);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
