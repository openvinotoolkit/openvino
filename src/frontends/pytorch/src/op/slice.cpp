// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <climits>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {
// Bound to use for an omitted start/end: `forward` when `step` is positive, `backward` when it is
// negative. Resolved at conversion time if `step` is known, otherwise selected at runtime.
Output<Node> default_bound(const NodeContext& context, const Output<Node>& step, int forward, int backward) {
    if (const auto step_const = ov::util::get_constant_from_source(step)) {
        const auto step_values = step_const->cast_vector<int64_t>();
        const bool is_backward = step_values.size() == 1 && step_values[0] < 0;
        return context.mark_node(v0::Constant::create(element::i32, Shape{1}, {is_backward ? backward : forward}));
    }
    auto forward_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {forward}));
    auto backward_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {backward}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, step));
    auto is_backward = context.mark_node(std::make_shared<v1::Less>(step, zero));
    return context.mark_node(std::make_shared<v1::Select>(is_backward, backward_const, forward_const));
}
}  // namespace

OutputVector translate_slice_common(const NodeContext& context,
                                    const size_t num_inputs,
                                    const bool stop_dynamic_rank_unsqueeze = true) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor(a))
    auto data = context.get_input(0);
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

    // An omitted start/end means the whole axis, and which end that is depends on the sign of step:
    // aten::slice.t allows a negative step (e.g. `shape[::-1]` in TorchScript), where the slice starts
    // at the last element and stops past the first one.
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
        start = default_bound(context, step, 0, -1);
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
        end = default_bound(context, step, INT_MAX, INT_MIN);
    }

    if (const auto complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr())) {
        auto rank = std::get<1>(get_shape_rank(context, data, true));
        dim = normalize_axis(context, dim, rank);
        data = complex->get_input_source_output(0);
        Output<Node> slice = context.mark_node(std::make_shared<v8::Slice>(data, start, end, step, dim));
        return {context.mark_node(std::make_shared<ComplexTypeMark>(slice, slice.get_element_type()))};
    } else {
        return {context.mark_node(std::make_shared<v8::Slice>(data, start, end, step, dim))};
    }
};

OutputVector translate_slice(const NodeContext& context) {
    return translate_slice_common(context, context.get_input_size());
};

OutputVector translate_slice_fx(const NodeContext& context) {
    // slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
    // FX version of slice have the inputs in the same order as it has 5 inputs, even if it has less than 5 inputs
    return translate_slice_common(context, 5, false);
};

}  // namespace ov::frontend::pytorch::op
