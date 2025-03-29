// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_one_hot(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    // aten::one_hot works on LongTensor which means we need to convert all inputs to i64
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::i64));
    auto on_value = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto zero_value = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    Output<Node> num_classes;
    if (context.input_is_none(1)) {
        num_classes = context.mark_node(v0::Constant::create(element::i64, Shape{}, {-1}));
    } else {
        num_classes = context.get_input(1);
        num_classes = context.mark_node(std::make_shared<v0::Convert>(num_classes, element::i64));
    }
    auto one = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto greater = context.mark_node(std::make_shared<v1::Greater>(num_classes, zero_value));
    auto axes = get_axes_range(context, 0);
    auto max_class = context.mark_node(std::make_shared<v1::ReduceMax>(x, axes));
    max_class = context.mark_node(std::make_shared<v1::Add>(max_class, one));
    num_classes = context.mark_node(std::make_shared<v1::Select>(greater, num_classes, max_class));
    return {context.mark_node(std::make_shared<v1::OneHot>(x, num_classes, on_value, zero_value, -1))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
