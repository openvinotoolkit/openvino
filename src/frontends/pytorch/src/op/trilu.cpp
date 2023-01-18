// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

namespace base {

OutputVector translate_base_triu_tril(NodeContext& context, bool upper) {
    auto input_tensor = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(input_tensor));
    auto zero = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {0}));
    auto one = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {1}));
    auto minus_one = context.mark_node(opset10::Constant::create(element::i32, Shape{}, {-1}));
    auto minus_two = context.mark_node(opset10::Constant::create(element::i32, Shape{}, {-2}));
    const auto m = context.mark_node(std::make_shared<opset10::Gather>(input_shape, minus_one, zero));
    const auto n = context.mark_node(std::make_shared<opset10::Gather>(input_shape, minus_two, zero));
    auto horizontal_range = context.mark_node(std::make_shared<opset10::Range>(zero, m, one, element::i64));
    horizontal_range = context.mark_node(std::make_shared<opset10::Unsqueeze>(horizontal_range, zero));
    Output<Node> vertical_range;
    if (!context.input_is_none(1)) {
        auto diagonal = context.get_input(1);
        diagonal = context.mark_node(std::make_shared<opset10::Convert>(diagonal, element::i64));
        auto stop = context.mark_node(std::make_shared<opset10::Add>(n, diagonal));
        vertical_range = context.mark_node(std::make_shared<opset10::Range>(diagonal, stop, one, element::i64));
    } else {
        vertical_range = context.mark_node(std::make_shared<opset10::Range>(zero, n, one, element::i64));
    }
    vertical_range = context.mark_node(std::make_shared<opset10::Unsqueeze>(vertical_range, one));

    Output<Node> mask;
    if (upper) {
        mask = context.mark_node(std::make_shared<opset10::GreaterEqual>(horizontal_range, vertical_range));
    } else {
        mask = context.mark_node(std::make_shared<opset10::LessEqual>(horizontal_range, vertical_range));
    }

    zero = context.mark_node(std::make_shared<opset10::ConvertLike>(zero, input_tensor));

    return {context.mark_node(std::make_shared<opset10::Select>(mask, input_tensor, zero))};
}
};  // namespace base

OutputVector translate_triu(NodeContext& context) {
    return base::translate_base_triu_tril(context, true);
};

OutputVector translate_tril(NodeContext& context) {
    return base::translate_base_triu_tril(context, false);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov