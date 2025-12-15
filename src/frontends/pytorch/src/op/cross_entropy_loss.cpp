// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/not_equal.hpp"


#include "utils.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {



int64_t resolve_reduction(const NodeContext& context) {
    int64_t reduction = 1;
    if (context.get_input_size() > 3 && !context.input_is_none(3)) {
        auto reduction_type = context.get_input_type(3);
        if (reduction_type.is<type::Str>()) {
            auto reduction_str = context.const_input<std::string>(3);
            if (reduction_str == "none") {
                reduction = 0;
            } else if (reduction_str == "mean") {
                reduction = 1;
            } else if (reduction_str == "sum") {
                reduction = 2;
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false,
                                            "Unsupported reduction mode for aten::cross_entropy_loss: ",
                                            reduction_str);
            }
        } else {
            reduction = context.const_input<int64_t>(3);
        }
    }
    return reduction;
}
}  // namespace

OutputVector translate_cross_entropy_loss(const NodeContext& context) {
    num_inputs_check(context, 2, 6);

    auto logits = get_input_with_floating_type(context, 0);
    auto target = context.get_input(1);

    const auto rank = logits.get_partial_shape().rank();
    PYTORCH_OP_CONVERSION_CHECK(rank.is_static() && rank.get_length() >= 1,
                                "aten::cross_entropy_loss expects input rank >= 1");

    if (target.get_element_type() != element::i64) {
        target = context.mark_node(std::make_shared<v0::Convert>(target, element::i64));
    }

    const int64_t class_axis = (rank.get_length() == 1) ? 0 : 1;

    auto log_probs = context.mark_node(
        std::make_shared<v5::LogSoftmax>(logits, class_axis));

    // ---------- weight ----------
    bool has_weight = context.get_input_size() > 2 && !context.input_is_none(2);
    Output<Node> sample_weight;

    if (has_weight) {
        auto weight = context.get_input(2);
        weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, logits));

        auto axis0 = context.mark_node(
            v0::Constant::create(element::i32, Shape{}, {0}));

        sample_weight = context.mark_node(
            std::make_shared<v8::Gather>(weight, target, axis0));
    }

    // ---------- ignore_index ----------
    bool use_ignore = false;
    int64_t ignore_index = -100;

    if (context.get_input_size() > 4 && !context.input_is_none(4)) {
        ignore_index = context.const_input<int64_t>(4);
        use_ignore = (ignore_index != -100);
    }

    Output<Node> valid_mask_f;

    if (use_ignore) {
        auto ignore_const = context.mark_node(
            v0::Constant::create(element::i64, Shape{}, {ignore_index}));

        auto valid_mask = context.mark_node(
            std::make_shared<v1::NotEqual>(target, ignore_const));

        valid_mask_f = context.mark_node(
            std::make_shared<v0::Convert>(valid_mask, logits.get_element_type()));
    }

    // ---------- one-hot ----------
    auto shape = context.mark_node(
        std::make_shared<v3::ShapeOf>(log_probs, element::i32));

    auto class_axis_const = context.mark_node(
        v0::Constant::create(element::i32, Shape{1}, {class_axis}));

    auto axis0 = context.mark_node(
        v0::Constant::create(element::i32, Shape{}, {0}));

    auto class_dim = context.mark_node(
        std::make_shared<v8::Gather>(shape, class_axis_const, axis0));

    const auto loss_et = logits.get_element_type();

    auto on_value = context.mark_node(
        v0::Constant::create(loss_et, Shape{}, {1.0f}));
    auto off_value = context.mark_node(
        v0::Constant::create(loss_et, Shape{}, {0.0f}));

    auto one_hot = context.mark_node(
        std::make_shared<v1::OneHot>(
            target, class_dim, on_value, off_value, class_axis));

    if (use_ignore) {
        one_hot = context.mark_node(
            std::make_shared<v1::Multiply>(one_hot, valid_mask_f));
    }

    // ---------- loss ----------
    auto loss = context.mark_node(
        std::make_shared<v1::Multiply>(log_probs, one_hot));

    auto reduce_axis = context.mark_node(
        v0::Constant::create(element::i32, Shape{1}, {class_axis}));

    loss = context.mark_node(
        std::make_shared<v0::Negative>(
            context.mark_node(
                std::make_shared<v1::ReduceSum>(loss, reduce_axis, false))));

    if (has_weight) {
        if (use_ignore) {
            sample_weight = context.mark_node(
                std::make_shared<v1::Multiply>(sample_weight, valid_mask_f));
        }
        loss = context.mark_node(
            std::make_shared<v1::Multiply>(loss, sample_weight));
    }

    // ---------- reduction ----------
    const int64_t reduction = resolve_reduction(context);

    if (reduction == 0) {  // none
        return {loss};
    }

    auto axes = get_node_axes_range(context, loss);

    if (reduction == 2) {  // sum
        return {context.mark_node(
            std::make_shared<v1::ReduceSum>(loss, axes, false))};
    }

    // mean
    auto loss_sum = context.mark_node(
        std::make_shared<v1::ReduceSum>(loss, axes, false));

    if (has_weight) {
        auto weight_sum = context.mark_node(
            std::make_shared<v1::ReduceSum>(sample_weight, axes, false));
        return {context.mark_node(
            std::make_shared<v1::Divide>(
                loss_sum,
                context.mark_node(
                    std::make_shared<v1::ConvertLike>(weight_sum, loss_sum))))};
    }

    auto count = use_ignore
        ? context.mark_node(std::make_shared<v1::ReduceSum>(valid_mask_f, axes, false))
        : numel(context, loss, element::i32);

    auto count_f = context.mark_node(
        std::make_shared<v1::ConvertLike>(count, loss_sum));

    return {context.mark_node(
        std::make_shared<v1::Divide>(loss_sum, count_f))};
}


}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov