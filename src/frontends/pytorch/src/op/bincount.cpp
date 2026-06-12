// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bincount(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::i32));

    Output<Node> weights;
    if (!context.input_is_none(1)) {
        weights = context.get_input(1);
    } else {
        weights = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    }

    Output<Node> minlength;
    if (!context.input_is_none(2)) {
        minlength = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::i32));
    } else {
        minlength = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    }

    auto reduce_axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto max_val = context.mark_node(std::make_shared<v1::ReduceMax>(input, reduce_axis, false));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto max_plus_one = context.mark_node(std::make_shared<v1::Add>(max_val, one));
    auto size = context.mark_node(std::make_shared<v1::Maximum>(max_plus_one, minlength));

    return common_translators::translate_bincount_common(context, input, size, weights);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
