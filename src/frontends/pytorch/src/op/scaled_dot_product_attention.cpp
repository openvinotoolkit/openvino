// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_scaled_dot_product_attention(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False, float scale=None)
    num_inputs_check(context, 6, 7);
    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);

    auto is_causal = context.const_input<bool>(5);
    OutputVector inputs = {query, key, value};  // mandatory inputs

    if (!context.input_is_none(3))
        inputs.push_back(context.get_input(3));
    else if (!context.input_is_none(6)) {
        // need to fill a gap in inputs with scalar 0 to be able to pass one extra input after that
        auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(zero, query)));
    }
    if (!context.input_is_none(6))
        inputs.push_back(context.get_input(6));

    return {context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov