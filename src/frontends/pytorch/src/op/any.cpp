// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> translate_any_common(const NodeContext& context,
                                  const Output<Node>& x,
                                  const Output<Node>& axes,
                                  bool keepdims) {
    element::Type output_dtype = element::boolean;
    if (x.get_element_type() == element::u8) {
        output_dtype = element::u8;
    }
    auto x_bool = x;
    if (x.get_element_type() != element::boolean) {
        x_bool = context.mark_node(std::make_shared<v0::Convert>(x_bool, element::boolean));
    }

    const auto all_nonzero = context.mark_node(std::make_shared<v1::ReduceLogicalOr>(x_bool, axes, keepdims));
    return context.mark_node(std::make_shared<v0::Convert>(all_nonzero, output_dtype));
};
}  // namespace

OutputVector translate_any(const NodeContext& context) {
    num_inputs_check(context, 1, 4);
    auto input_tensor = context.get_input(0);

    auto num_inputs = context.get_input_size();
    size_t out_id;

    bool keep_dims = false;
    ov::Output<ov::Node> axes;
    if (num_inputs < 3) {
        axes = get_axes_range(context, 0);
        out_id = 1;
    } else {
        const auto dim = context.const_input<int64_t>(1);
        axes = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {dim}));
        if (!context.input_is_none(2)) {
            keep_dims = context.const_input<bool>(2);
        }
        out_id = 3;
    }

    const auto& result = translate_any_common(context, input_tensor, axes, keep_dims);
    if (!context.input_is_none(out_id)) {
        context.mutate_input(out_id, result);
    }
    return {result};
};

OutputVector translate_any_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);

    Output<Node> dims;
    if (!context.input_is_none(1)) {
        dims = context.get_input(1);
    } else {
        dims = get_axes_range(context, 0);
    }
    bool keep_dims = false;
    if (!context.input_is_none(2))
        keep_dims = context.const_input<bool>(2);
    const auto& any = translate_any_common(context, x, dims, keep_dims);
    return {any};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
