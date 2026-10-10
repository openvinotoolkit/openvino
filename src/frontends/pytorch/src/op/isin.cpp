// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {
using namespace ov::op;

OutputVector translate_isin(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto elements = context.get_input(0);
    auto tests = context.get_input(1);
    align_eltwise_input_types(context, elements, tests);
    auto flat_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto flat_elements = context.mark_node(std::make_shared<v1::Reshape>(elements, flat_shape, false));
    auto flat_tests = context.mark_node(std::make_shared<v1::Reshape>(tests, flat_shape, false));
    auto expanded = context.mark_node(std::make_shared<v0::Unsqueeze>(flat_elements, axis));
    auto equal = context.mark_node(std::make_shared<v1::Equal>(expanded, flat_tests));
    Output<Node> result = context.mark_node(std::make_shared<v1::ReduceLogicalOr>(equal, axis, false));
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(elements));
    result = context.mark_node(std::make_shared<v1::Reshape>(result, shape, false));
    bool invert = context.has_attribute("invert") ? context.get_attribute<bool>("invert") : false;
    if (!context.input_is_none(3)) {
        invert = context.const_input<bool>(3);
    }
    if (invert) {
        result = context.mark_node(std::make_shared<v1::LogicalNot>(result));
    }
    if (!context.input_is_none(4)) {
        context.mutate_input(4, result);
    }
    return {result};
}
}  // namespace ov::frontend::pytorch::op
