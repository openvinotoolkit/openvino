// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_group_norm(NodeContext& context) {
    auto data = context.get_input(0);
    auto num_groups = context.const_input<int64_t>(1);
    auto eps = context.const_input<double>(4);
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(data, element::i64));
    auto shape = context.mark_node(
        std::make_shared<opset8::Constant>(element::i64, Shape({3}), std::vector<int64_t>{0, num_groups, -1}));
    auto reshaped_input = context.mark_node(std::make_shared<opset8::Reshape>(data, shape, true));
    auto value_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(reshaped_input, element::i64));
    const auto reduction_axes = std::make_shared<opset8::Range>(
        opset8::Constant::create(element::i64, {}, {2}),
        std::make_shared<opset8::Squeeze>(std::make_shared<opset8::ShapeOf>(value_shape, element::i64)),
        opset8::Constant::create(element::i64, {}, {1}),
        element::i64);
    auto reshaped_norm = context.mark_node(
        std::make_shared<opset8::MVN>(reshaped_input, reduction_axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT));
    auto norm = context.mark_node(std::make_shared<opset8::Reshape>(reshaped_norm, input_shape, true));
    auto input_rank = std::make_shared<opset8::Squeeze>(std::make_shared<opset8::ShapeOf>(input_shape, element::i64));
    auto skip_last = std::make_shared<opset8::Subtract>(input_rank, opset8::Constant::create(element::i64, {}, {1}));
    auto axes = std::make_shared<opset8::Range>(opset8::Constant::create(element::i64, {}, {1}),
                                                skip_last,
                                                opset8::Constant::create(element::i64, {}, {1}),
                                                element::i64);
    if (!context.input_is_none(2)) {
        auto weights = context.get_input(2);
        weights = context.mark_node(std::make_shared<opset8::Unsqueeze>(weights, axes));
        norm = context.mark_node(std::make_shared<opset8::Multiply>(norm, weights));
    }
    if (!context.input_is_none(3)) {
        auto bias = context.get_input(3);
        bias = context.mark_node(std::make_shared<opset8::Unsqueeze>(bias, axes));
        norm = context.mark_node(std::make_shared<opset8::Add>(norm, bias));
    }
    return {norm};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov