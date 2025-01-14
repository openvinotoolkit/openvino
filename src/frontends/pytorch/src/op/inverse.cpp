// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/inverse.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_inverse(const NodeContext& context) {
    // aten::inverse(Tensor self) -> Tensor
    // aten::inverse(Tensor self, *, Tensor out) -> Tensor
    num_inputs_check(context, 1, 2);
    const auto input_tensor = context.get_input(0);
    const auto input_type = input_tensor.get_element_type();

    ov::Output<ov::Node> result;
    if (input_type == element::bf16 || input_type == element::f16 || input_type == element::f32) {
        result = context.mark_node(std::make_shared<ov::op::v14::Inverse>(input_tensor, false));
    } else {
        const auto converted_input =
            context.mark_node(std::make_shared<ov::op::v0::Convert>(input_tensor, element::f32));
        auto inverse = context.mark_node(std::make_shared<ov::op::v14::Inverse>(converted_input, false));
        result = context.mark_node(std::make_shared<ov::op::v0::Convert>(inverse, input_type));
    }

    if (!context.input_is_none(1)) {
        context.mutate_input(1, result);
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
