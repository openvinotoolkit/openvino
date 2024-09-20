// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "helper_ops/slice_assign.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_slice_scatter_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto src = context.get_input(1);
    auto axis_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    ov::Output<ov::Node> dim;
    if (!context.input_is_none(2)) {
        dim = context.get_input(2);
        if (dim.get_partial_shape().rank().is_static() && dim.get_partial_shape().rank().get_length() == 0) {
            dim = context.mark_node(std::make_shared<v0::Unsqueeze>(dim, axis_0));
        }
    } else {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    }
    ov::Output<ov::Node> start;
    if (!context.input_is_none(3)) {
        start = context.get_input(3);
        if (start.get_partial_shape().rank().is_static() && start.get_partial_shape().rank().get_length() == 0) {
            start = context.mark_node(std::make_shared<v0::Unsqueeze>(start, axis_0));
        }
    } else {
        start = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    }
    ov::Output<ov::Node> end;
    if (!context.input_is_none(4)) {
        end = context.get_input(4);
        if (end.get_partial_shape().rank().is_static() && end.get_partial_shape().rank().get_length() == 0) {
            end = context.mark_node(std::make_shared<v0::Unsqueeze>(end, axis_0));
        }
    } else {
        end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {INT_MAX}));
    }
    ov::Output<ov::Node> step;
    if (!context.input_is_none(5)) {
        step = context.get_input(5);
        if (step.get_partial_shape().rank().is_static() && step.get_partial_shape().rank().get_length() == 0) {
            step = context.mark_node(std::make_shared<v0::Unsqueeze>(step, axis_0));
        }
    } else {
        step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    }
    return {context.mark_node(std::make_shared<SliceAssign>(input, src, start, end, step, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
