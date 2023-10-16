// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_scaled_dot_product_attention(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False)
    num_inputs_check(context, 6, 6);
    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);
    std::shared_ptr<Node> node;

    auto is_causal = context.const_input<bool>(5);
    if (context.input_is_none(3)) {
        node = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, is_causal);
    } else {
        node = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, context.get_input(3), is_causal);
    }
    return {context.mark_node(node)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov