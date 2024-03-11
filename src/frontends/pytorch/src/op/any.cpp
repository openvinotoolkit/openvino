// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_any_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto num_inputs = context.get_input_size();
    bool keep_dims = false;

    auto const_minus_one = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto flatten_source = context.mark_node(std::make_shared<ov::op::v1::Reshape>(x, const_minus_one, false));
    auto const_zero = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {0}));
    auto zero = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{}, {1}));
    auto rank = std::get<1>(get_shape_rank(context, x, true));
    auto axes = context.mark_node(std::make_shared<ov::op::v0::Range>(zero, rank, one));
    auto any = context.mark_node(std::make_shared<ov::op::v1::ReduceLogicalOr>(x, axes, keep_dims));
    return {any};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
