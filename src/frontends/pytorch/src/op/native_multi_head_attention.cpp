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

using namespace ov::op;

OutputVector translate_native_multi_head_attention(const NodeContext& context) {
    /* pytorch.org/cppdocs/api/function_namespaceat_1aa4f72ac82c15c7aeef274332b25a543b.html
    aten::_native_multi_head_attention(
        Tensor query, 
        Tensor key, 
        Tensor value, 
        int64 embed_dim, 
        int64 num_head, 
        Tensor qkv_weight, 
        Tensor qkv_bias, 
        Tensor proj_weight, 
        Tensor proj_bias, 
        Optional[Tensor] mask = None, 
        bool need_weights = true, 
        bool average_attn_weights = true, 
        Optional[int64] mask_type = None 
    )
    */
    num_inputs_check(context, 13, 13);
    const auto query = context.get_input(0);
    const auto key = context.get_input(1);
    const auto value = context.get_input(2);
    const auto embed_dim = context.const_input<int64_t>(3);
    const auto num_heads = context.const_input<int64_t>(4);
    const auto qkv_weight = context.get_input(5);
    const auto qkv_bias = context.get_input(6);
    const auto proj_weight = context.get_input(7);
    const auto proj_bias = context.get_input(8);
    const auto need_weights = context.const_input<bool>(10);
    const auto average_attn_weights = context.const_input<bool>(11);

    const auto zero = context.mark_node(opset10::Constant::create(element::i32, Shape{}, {0}));
    const auto one = context.mark_node(opset10::Constant::create(element::i32, Shape{}, {1}));
    const auto two = context.mark_node(opset10::Constant::create(element::i32, Shape{}, {2}));

    const auto qkv_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(query));

    const auto scale_one = context.mark_node(std::make_shared<opset10::ConvertLike>(one, query));
    const auto scale_dim = context.mark_node(std::make_shared<opset10::ConvertLike>(embed_dim, query));
    const auto scale_dim_sqrt = context.mark_node(std::make_shared<opset10::Sqrt>(scale_dim));
    const auto scale = context.mark_node(std::make_shared<opset10::Divide>(scale_one, scale_dim_sqrt));

    const auto transpose_dims = context.mark_node(std::make_shared<opset10::Concat>(OutputVector{zero, two, one}, 0));
    const auto key_transpose = context.mark_node(std::make_shared<opset10::Transpose>(key, transpose_dims));
    const auto query_key_transpose_dot_product = context.mark_node(std::make_shared<opset10::MatMul>(query, key_transpose));
    const auto scaled_dot_product = context.mark_node(std::make_shared<opset10::Multiply>(query_key_transpose_dot_product, scale));
    const auto scaled_dot_product_softmax = context.mark_node(std::make_shared<opset10::Softmax>(scaled_dot_product, -1));
    const auto scaled_dot_product_attention = context.mark_node(std::make_shared<opset10::MatMul>(scaled_dot_product_softmax, value));
};  

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov