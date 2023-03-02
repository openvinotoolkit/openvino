// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_slice(NodeContext& context) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor(a))
    ov::Output<ov::Node> dim;
    int start_idx;
    int end_idx;
    int step_idx;
    auto axis_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    if (context.get_input_size() == 5) {
        dim = context.get_input(1);
        if (dim.get_partial_shape().rank().is_dynamic() || dim.get_partial_shape().rank().get_length() == 0) {
            dim = context.mark_node(std::make_shared<v0::Unsqueeze>(dim, axis_0));
        }
        start_idx = 2;
        end_idx = 3;
        step_idx = 4;
    } else if (context.get_input_size() == 4) {
        start_idx = 1;
        end_idx = 2;
        step_idx = 3;
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Slice must have either 4 or 5 inputs.");
    }
    // TODO: support default start/end with negative step
    ov::Output<ov::Node> start;
    if (!context.input_is_none(start_idx)) {
        start = context.get_input(start_idx);
        if (start.get_partial_shape().rank().is_dynamic() || start.get_partial_shape().rank().get_length() == 0) {
            start = context.mark_node(std::make_shared<v0::Unsqueeze>(start, axis_0));
        }
    } else {
        start = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    }

    ov::Output<ov::Node> end;
    if (!context.input_is_none(end_idx)) {
        end = context.get_input(end_idx);
        if (end.get_partial_shape().rank().is_dynamic() || end.get_partial_shape().rank().get_length() == 0) {
            end = context.mark_node(std::make_shared<v0::Unsqueeze>(end, axis_0));
        }
    } else {
        end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {INT_MAX}));
    }
    ov::Output<ov::Node> step;
    if (!context.input_is_none(step_idx)) {
        step = context.get_input(step_idx);
        if (step.get_partial_shape().rank().is_dynamic() || step.get_partial_shape().rank().get_length() == 0) {
            step = context.mark_node(std::make_shared<v0::Unsqueeze>(step, axis_0));
        }
    } else {
        step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    }
    return {context.mark_node(std::make_shared<v8::Slice>(context.get_input(0), start, end, step, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov