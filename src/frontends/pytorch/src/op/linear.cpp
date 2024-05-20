// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_linear(const NodeContext& context) {
    // schema: aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto weight = context.get_input(1);
    if (weight.get_element_type() == element::f16 || weight.get_element_type() == element::bf16) {
        // In case of patched linear it can have mixed fp16/bf16 and fp32 input type.
        // In other cases these conversion is not required.
        weight = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(weight, x));
    }
    auto matmul = context.mark_node(std::make_shared<ov::op::v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            // Same reason as for weight.
            bias = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<ov::op::v1::Add>(matmul, bias));
    }
    return {matmul};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov