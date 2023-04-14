// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_all(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    const auto input_tensor = context.get_input(0);

    element::Type output_dtype;
    if (input_tensor.get_element_type() == element::u8) {
        output_dtype = element::u8;
    } else {
        output_dtype = element::boolean;
    }

    bool keep_dims;
    ov::Output<ov::Node> axes;
    if (context.get_input_size() == 1) {
        keep_dims = false;
        axes = get_axes_range(context, 0);
    } else {
        const auto dim = context.const_input<int64_t>(1);
        keep_dims = !context.input_is_none(2) ? context.const_input<bool>(2) : false;
        axes = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {dim}));
    }

    const auto all_nonzero = context.mark_node(std::make_shared<opset10::ReduceProd>(input_tensor, axes, keep_dims));
    return {context.mark_node(std::make_shared<opset10::Convert>(all_nonzero, output_dtype))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
