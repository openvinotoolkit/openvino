// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cross_entropy_loss(const NodeContext& context) {
    // aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None,
    //                          int reduction=1, SymInt ignore_index=-100, float label_smoothing=0.) -> Tensor
    num_inputs_check(context, 2, 6);

    auto logits = get_input_with_floating_type(context, 0);
    auto target = context.get_input(1);

    if (target.get_element_type() != element::i64) {
        target = context.mark_node(std::make_shared<v0::Convert>(target, element::i64));
    }

    const auto logits_shape = logits.get_partial_shape();
    const auto rank = logits_shape.rank();
    PYTORCH_OP_CONVERSION_CHECK(rank.is_dynamic() || rank.get_length() >= 1,
                                "aten::cross_entropy_loss expects input rank >= 1");

    int64_t class_axis = 1;
    bool rank_is_static = rank.is_static();
    if (rank_is_static && rank.get_length() == 1) {
        class_axis = 0;
    }

    // log_softmax along class axis
    auto log_probs = context.mark_node(std::make_shared<v5::LogSoftmax>(logits, class_axis));

    // ---------- weight ----------
    bool has_weight = context.get_input_size() > 2 && !context.input_is_none(2);
    Output<Node> sample_weight;

    if (has_weight) {
        auto weight = context.get_input(2);
        weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, logits));

        auto axis_const = context.mark_node(
            v0::Constant::create(element::i32, Shape{}, {0}));

        sample_weight = context.mark_node(
            std::make_shared<v8::Gather>(weight, target, axis_const));
    }

    // reduction (input 3, default 1=mean)
    int64_t reduction = 1;  // 0=none, 1=mean, 2=sum
    if (context.get_input_size() > 3 && !context.input_is_none(3)) {
        reduction = context.const_input<int64_t>(3);
    }

    // ignore_index (input 4, default -100)
    int64_t ignore_index = -100;
    bool use_ignore = false;
    if (context.get_input_size() > 4 && !context.input_is_none(4)) {
        ignore_index = context.const_input<int64_t>(4);
        use_ignore = true;
    }

    // label_smoothing (input 5) - currently not implemented
    if (context.get_input_size() > 5 && !context.input_is_none(5)) {
        float label_smoothing = context.const_input<float>(5);
        PYTORCH_OP_CONVERSION_CHECK(label_smoothing == 0.0f,
                                    "aten::cross_entropy_loss: label_smoothing is not yet supported");
    }

    // ---------- Gather log probabilities using GatherElements ----------
    // 2D: log_probs is (N, C), target is (N,)
    // 1D: log_probs is (C,), target is scalar

    Output<Node> target_expanded;
    if (rank_is_static && rank.get_length() == 1) {
        target_expanded = target;
    } else {
        // 2D+: add a dimension at class_axis
        // target shape: (N,) or (N, d1, d2, ...) -> (N, 1) or (N, 1, d1, d2, ...)
        auto axis_const = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {class_axis}));
        target_expanded = context.mark_node(
            std::make_shared<v0::Unsqueeze>(target, axis_const));
    }

    auto gathered_log_probs = context.mark_node(
        std::make_shared<v6::GatherElements>(log_probs, target_expanded, class_axis));

    if (!rank_is_static || rank.get_length() > 1) {
        auto axis_const = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {class_axis}));
        gathered_log_probs = context.mark_node(
            std::make_shared<v0::Squeeze>(gathered_log_probs, axis_const));
    }

    auto loss = context.mark_node(std::make_shared<v0::Negative>(gathered_log_probs));

    Output<Node> valid_mask_float;
    if (use_ignore) {
        auto ignore_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {ignore_index}));
        auto valid_mask = context.mark_node(std::make_shared<v1::NotEqual>(target, ignore_const));
        valid_mask_float = context.mark_node(std::make_shared<v1::ConvertLike>(valid_mask, loss));

        loss = context.mark_node(std::make_shared<v1::Multiply>(loss, valid_mask_float));
    }

    // sample weights
    if (has_weight) {
        if (use_ignore) {
            sample_weight = context.mark_node(
                std::make_shared<v1::Multiply>(sample_weight, valid_mask_float));
        }
        loss = context.mark_node(
            std::make_shared<v1::Multiply>(loss, sample_weight));
    }

    // ---------- reduction ----------
    if (reduction == 0) {
        // none - return per-sample loss
        return {loss};
    }

    // Get all axes for reduction
    auto all_axes = get_node_axes_range(context, loss);

    if (reduction == 2) {
        // sum
        return {context.mark_node(std::make_shared<v1::ReduceSum>(loss, all_axes, false))};
    }

    // reduction == 1: mean
    auto loss_sum = context.mark_node(std::make_shared<v1::ReduceSum>(loss, all_axes, false));

    if (has_weight) {
        auto weight_sum = context.mark_node(
            std::make_shared<v1::ReduceSum>(sample_weight, all_axes, false));
        weight_sum = context.mark_node(
            std::make_shared<v1::ConvertLike>(weight_sum, loss_sum));
        return {context.mark_node(
            std::make_shared<v1::Divide>(loss_sum, weight_sum))};
    }

    Output<Node> count;
    if (use_ignore) {
        count = context.mark_node(
            std::make_shared<v1::ReduceSum>(valid_mask_float, all_axes, false));
    } else {
        count = numel(context, loss, element::i32);
    }

    count = context.mark_node(
        std::make_shared<v1::ConvertLike>(count, loss_sum));
    return {context.mark_node(
        std::make_shared<v1::Divide>(loss_sum, count))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
