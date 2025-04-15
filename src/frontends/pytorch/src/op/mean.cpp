// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_mean(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto x = context.get_input(0);
    auto num_inputs = context.get_input_size();
    bool keep_dims = false;
    Output<Node> axes;
    // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
    if (num_inputs == 2) {
        if (!context.input_is_none(1)) {
            x = apply_dtype(context, 1, x);
        }
        axes = get_axes_range(context, 0);
    } else {
        // aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        // aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) ->
        // Tensor(a!)
        axes = context.get_input(1);
        if (!context.input_is_none(2)) {
            keep_dims = context.const_input<bool>(2);
        }
        if (!context.input_is_none(3)) {
            x = apply_dtype(context, 3, x);
        }
    }
    auto mean = context.mark_node(std::make_shared<v1::ReduceMean>(x, axes, keep_dims));
    if (num_inputs == 5 && !context.input_is_none(4)) {
        context.mutate_input(4, mean);
    }
    return {mean};
};

OutputVector translate_mean_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 5);
    auto x = context.get_input(0);
    auto num_inputs = context.get_input_size();
    bool keep_dims = false;
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        x = context.mark_node(std::make_shared<v0::Convert>(x, dtype));
    }
    Output<Node> axes;
    if (num_inputs == 1) {
        axes = get_node_axes_range(context, x);
    } else if (num_inputs == 2) {
        axes = context.get_input(1);
    } else {
        axes = context.get_input(1);
        if (!context.input_is_none(2)) {
            keep_dims = context.const_input<bool>(2);
        }
        if (!context.input_is_none(3)) {
            x = apply_dtype(context, 3, x);
        }
    }
    auto mean = context.mark_node(std::make_shared<v1::ReduceMean>(x, axes, keep_dims));
    if (num_inputs == 5 && !context.input_is_none(4)) {
        context.mutate_input(4, mean);
    }
    return {mean};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov