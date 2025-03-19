// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <climits>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_slice_common(const NodeContext& context,
                                    const size_t num_inputs,
                                    const bool stop_dynamic_rank_unsqueeze = true) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor(a))
    ov::Output<ov::Node> dim;
    int start_idx;
    int end_idx;
    int step_idx;
    auto dims_1d_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    if (num_inputs == 5) {
        dim = context.get_input(1);
        if (dim.get_partial_shape().rank().is_dynamic() || dim.get_partial_shape().rank().get_length() == 0) {
            dim = context.mark_node(std::make_shared<v1::Reshape>(dim, dims_1d_shape, false));
            if (const auto axis_const = ov::util::get_constant_from_source(dim)) {
                dim = axis_const;
            }
        }
        start_idx = 2;
        end_idx = 3;
        step_idx = 4;
    } else if (num_inputs == 4) {
        start_idx = 1;
        end_idx = 2;
        step_idx = 3;
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Slice must have either 4 or 5 inputs.");
    }
    // TODO: support default start/end with negative step
    ov::Output<ov::Node> start;
    if (!context.input_is_none(start_idx)) {
        start = context.get_input(start_idx);
        if (start.get_partial_shape().rank().is_dynamic() || start.get_partial_shape().rank().get_length() == 0) {
            start = context.mark_node(std::make_shared<v1::Reshape>(start, dims_1d_shape, false));
        }
        if (const auto start_const = ov::util::get_constant_from_source(start)) {
            start = start_const;
        }
    } else {
        start = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    }

    ov::Output<ov::Node> end;
    if (!context.input_is_none(end_idx)) {
        end = context.get_input(end_idx);
        // TODO: Find a better way to solve the issue with dynamic ranks for "end"
        if ((stop_dynamic_rank_unsqueeze && end.get_partial_shape().rank().is_dynamic()) ||
            (!(end.get_partial_shape().rank().is_dynamic()) && end.get_partial_shape().rank().get_length() == 0)) {
            end = context.mark_node(std::make_shared<v1::Reshape>(end, dims_1d_shape, false));
        }
        if (const auto end_const = ov::util::get_constant_from_source(end)) {
            end = end_const;
        }
    } else {
        end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {INT_MAX}));
    }
    ov::Output<ov::Node> step;
    if (!context.input_is_none(step_idx)) {
        step = context.get_input(step_idx);
        if (step.get_partial_shape().rank().is_dynamic() || step.get_partial_shape().rank().get_length() == 0) {
            step = context.mark_node(std::make_shared<v1::Reshape>(step, dims_1d_shape, false));
        }
        if (const auto step_const = ov::util::get_constant_from_source(step)) {
            step = step_const;
        }
    } else {
        step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    }
    return {context.mark_node(std::make_shared<v8::Slice>(context.get_input(0), start, end, step, dim))};
};

OutputVector translate_slice(const NodeContext& context) {
    return translate_slice_common(context, context.get_input_size());
};

OutputVector translate_slice_fx(const NodeContext& context) {
    // slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
    // FX version of slice have the inputs in the same order as it has 5 inputs, even if it has less than 5 inputs
    return translate_slice_common(context, 5, false);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
