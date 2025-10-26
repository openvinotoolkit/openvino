// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <string>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

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

    auto rank = logits.get_partial_shape().rank();
    if (rank.is_static()) {
        PYTORCH_OP_CONVERSION_CHECK(rank.get_length() >= 2,
                                    "aten::cross_entropy_loss expects input rank >= 2 but got ",
                                    rank);
    }

    if (target.get_element_type() != element::i64) {
        target = context.mark_node(std::make_shared<v0::Convert>(target, element::i64));
    }

    auto log_probs = context.mark_node(std::make_shared<v5::LogSoftmax>(logits, 1));

    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(log_probs, element::i32));
    auto axis_zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto class_axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto class_dim = context.mark_node(std::make_shared<v8::Gather>(shape, class_axis, axis_zero));
    auto squeeze_axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto depth = context.mark_node(std::make_shared<v0::Squeeze>(class_dim, squeeze_axis));

    auto on_value = context.mark_node(v0::Constant::create(log_probs.get_element_type(), Shape{}, {1.0f}));
    auto off_value = context.mark_node(v0::Constant::create(log_probs.get_element_type(), Shape{}, {0.0f}));
    auto one_hot = context.mark_node(std::make_shared<v1::OneHot>(target, depth, on_value, off_value, 1));

    auto multiplied = context.mark_node(std::make_shared<v1::Multiply>(log_probs, one_hot));
    auto reduce_axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto gathered = context.mark_node(std::make_shared<v1::ReduceSum>(multiplied, reduce_axis, false));
    auto loss = context.mark_node(std::make_shared<v0::Negative>(gathered));

    Output<Node> sample_weight;
    const bool has_weight = context.get_input_size() > 2 && !context.input_is_none(2);
    if (has_weight) {
        auto weight = context.get_input(2);
        weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, loss));
        sample_weight = context.mark_node(std::make_shared<v8::Gather>(weight, target, axis_zero));
        loss = context.mark_node(std::make_shared<v1::Multiply>(loss, sample_weight));
    }

    const auto reduction = resolve_reduction(context);

    if (context.get_input_size() > 4 && !context.input_is_none(4)) {
        const auto ignore_index = context.const_input<int64_t>(4);
        PYTORCH_OP_CONVERSION_CHECK(ignore_index == -100,
                                    "aten::cross_entropy_loss with ignore_index=",
                                    ignore_index,
                                    " is not supported.");
    }

    if (context.get_input_size() > 5 && !context.input_is_none(5)) {
        const auto label_smoothing = context.const_input<double>(5);
        PYTORCH_OP_CONVERSION_CHECK(std::abs(label_smoothing) < 1e-8,
                                    "aten::cross_entropy_loss with label_smoothing=",
                                    label_smoothing,
                                    " is not supported.");
    }

    auto reduction_axes = get_node_axes_range(context, loss);
    Output<Node> result = loss;

    switch (reduction) {
    case 0:  // none
        break;
    case 1: {  // mean
        auto loss_sum = context.mark_node(std::make_shared<v1::ReduceSum>(loss, reduction_axes, false));
        if (has_weight) {
            auto weight_sum = context.mark_node(std::make_shared<v1::ReduceSum>(sample_weight, reduction_axes, false));
            auto weight_sum_cast = context.mark_node(std::make_shared<v1::ConvertLike>(weight_sum, loss_sum));
            result = context.mark_node(std::make_shared<v1::Divide>(loss_sum, weight_sum_cast));
        } else {
            auto count = numel(context, loss, element::i32);
            auto count_cast = context.mark_node(std::make_shared<v1::ConvertLike>(count, loss_sum));
            result = context.mark_node(std::make_shared<v1::Divide>(loss_sum, count_cast));
        }
        break;
    }
    case 2: {  // sum
        result = context.mark_node(std::make_shared<v1::ReduceSum>(loss, reduction_axes, false));
        break;
    }
    default:
        PYTORCH_OP_CONVERSION_CHECK(false,
                                    "Unsupported reduction value for aten::cross_entropy_loss: ",
                                    reduction);
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
