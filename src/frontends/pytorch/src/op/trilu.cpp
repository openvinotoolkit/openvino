// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector translate_base_triu_tril(const NodeContext& context, bool upper) {
    num_inputs_check(context, 1, 2);
    auto input_tensor = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input_tensor, element::i32));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto minus_two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-2}));
    const auto m = context.mark_node(std::make_shared<v7::Gather>(input_shape, minus_one, zero));
    const auto n = context.mark_node(std::make_shared<v7::Gather>(input_shape, minus_two, zero));
    auto horizontal_range = context.mark_node(std::make_shared<v4::Range>(zero, m, one, element::i32));
    horizontal_range = context.mark_node(std::make_shared<v0::Unsqueeze>(horizontal_range, zero));
    Output<Node> vertical_range;
    if (!context.input_is_none(1)) {
        auto diagonal = context.get_input(1);
        diagonal = context.mark_node(std::make_shared<v0::Convert>(diagonal, element::i32));
        auto stop = context.mark_node(std::make_shared<v1::Add>(n, diagonal));
        vertical_range = context.mark_node(std::make_shared<v4::Range>(diagonal, stop, one, element::i32));
    } else {
        vertical_range = context.mark_node(std::make_shared<v4::Range>(zero, n, one, element::i32));
    }
    vertical_range = context.mark_node(std::make_shared<v0::Unsqueeze>(vertical_range, one));

    Output<Node> mask;
    if (upper) {
        mask = context.mark_node(std::make_shared<v1::GreaterEqual>(horizontal_range, vertical_range));
    } else {
        mask = context.mark_node(std::make_shared<v1::LessEqual>(horizontal_range, vertical_range));
    }

    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, input_tensor));

    return {context.mark_node(std::make_shared<v1::Select>(mask, input_tensor, zero))};
}
};  // namespace

OutputVector translate_triu(const NodeContext& context) {
    return translate_base_triu_tril(context, true);
};

OutputVector translate_tril(const NodeContext& context) {
    return translate_base_triu_tril(context, false);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov