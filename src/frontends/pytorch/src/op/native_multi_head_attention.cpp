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
    const auto num_head = context.const_input<int64_t>(4);
    const auto qkv_weight = context.get_input(5);
    const auto qkv_bias = context.get_input(6);
    const auto proj_weight = context.get_input(7);
    const auto proj_bias = context.get_input(8);
    const auto need_weights = context.const_input<bool>(10);
    const auto average_attn_weights = context.const_input<bool>(11);

    const auto neg_one = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {-1}));
    const auto zero = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {0}));
    const auto one = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {1}));
    const auto two = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {2}));
    const auto three = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {3}));
    const auto ev = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {embed_dim}));
    const auto heads = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {num_head}));

    const auto neg_one_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {-1}));
    const auto zero_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {0}));
    const auto one_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    const auto two_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {2}));
    const auto three_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {3}));
    const auto heads_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {num_head}));

    const auto ev_1_slice_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {embed_dim}));
    const auto ev_2_slice_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {2 * embed_dim}));
    const auto ev_3_slice_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {3 * embed_dim}));

    const auto qkv_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(query));
    const auto batch_size = context.mark_node(std::make_shared<opset10::Gather>(qkv_shape, zero_1d, zero_1d));
    const auto seq_size = context.mark_node(std::make_shared<opset10::Gather>(qkv_shape, one_1d, zero_1d));
    const auto embed_div_heads = context.mark_node(std::make_shared<opset10::Divide>(ev, heads, true));

    const auto query_proj_weight =
        context.mark_node(std::make_shared<opset10::Slice>(qkv_weight, zero_1d, ev_1_slice_1d, one_1d, zero_1d));
    const auto key_proj_weight =
        context.mark_node(std::make_shared<opset10::Slice>(qkv_weight, ev_1_slice_1d, ev_2_slice_1d, one_1d, zero_1d));
    const auto value_proj_weight =
        context.mark_node(std::make_shared<opset10::Slice>(qkv_weight, ev_2_slice_1d, ev_3_slice_1d, one_1d, zero_1d));
    const auto query_proj_bias =
        context.mark_node(std::make_shared<opset10::Slice>(qkv_bias, zero_1d, ev_1_slice_1d, one_1d, zero_1d));
    const auto key_proj_bias =
        context.mark_node(std::make_shared<opset10::Slice>(qkv_bias, ev_1_slice_1d, ev_2_slice_1d, one_1d, zero_1d));
    const auto value_proj_bias =
        context.mark_node(std::make_shared<opset10::Slice>(qkv_bias, ev_2_slice_1d, ev_3_slice_1d, one_1d, zero_1d));

    const auto query_weighted = context.mark_node(std::make_shared<opset10::MatMul>(query, query_proj_weight));
    const auto key_weighted = context.mark_node(std::make_shared<opset10::MatMul>(key, key_proj_weight));
    const auto value_weighted = context.mark_node(std::make_shared<opset10::MatMul>(value, value_proj_weight));

    const auto query_biased = context.mark_node(std::make_shared<opset10::Add>(query_weighted, query_proj_bias));
    const auto key_biased = context.mark_node(std::make_shared<opset10::Add>(key_weighted, key_proj_bias));
    const auto value_biased = context.mark_node(std::make_shared<opset10::Add>(value_weighted, value_proj_bias));

    const auto qkv_reshape_dims = context.mark_node(
        std::make_shared<opset10::Concat>(OutputVector{batch_size, seq_size, heads_1d, neg_one_1d}, 0));
    const auto qkv_transpose_dims =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{zero_1d, two_1d, one_1d, three_1d}, 0));

    const auto query_reshaped =
        context.mark_node(std::make_shared<opset10::Reshape>(query_biased, qkv_reshape_dims, false));
    const auto key_reshaped =
        context.mark_node(std::make_shared<opset10::Reshape>(key_biased, qkv_reshape_dims, false));
    const auto value_reshaped =
        context.mark_node(std::make_shared<opset10::Reshape>(value_biased, qkv_reshape_dims, false));

    const auto query_transposed =
        context.mark_node(std::make_shared<opset10::Transpose>(query_reshaped, qkv_transpose_dims));
    const auto key_transposed =
        context.mark_node(std::make_shared<opset10::Transpose>(key_reshaped, qkv_transpose_dims));
    const auto value_transposed =
        context.mark_node(std::make_shared<opset10::Transpose>(value_reshaped, qkv_transpose_dims));

    const auto scale_one = context.mark_node(std::make_shared<opset10::ConvertLike>(one, query_transposed));
    const auto scale_dim = context.mark_node(std::make_shared<opset10::ConvertLike>(ev, query_transposed));
    const auto scale_dim_sqrt = context.mark_node(std::make_shared<opset10::Sqrt>(scale_dim));
    const auto scale = context.mark_node(std::make_shared<opset10::Divide>(scale_one, scale_dim_sqrt));

    const auto transpose_dims =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{zero_1d, one_1d, three_1d, two_1d}, 0));
    const auto key_transpose = context.mark_node(std::make_shared<opset10::Transpose>(key_transposed, transpose_dims));
    const auto query_key_transpose_dot_product =
        context.mark_node(std::make_shared<opset10::MatMul>(query_transposed, key_transpose));
    const auto scaled_dot_product =
        context.mark_node(std::make_shared<opset10::Multiply>(query_key_transpose_dot_product, scale));
    const auto scaled_dot_product_softmax =
        context.mark_node(std::make_shared<opset10::Softmax>(scaled_dot_product, -1));
    const auto scaled_dot_product_attention =
        context.mark_node(std::make_shared<opset10::MatMul>(scaled_dot_product_softmax, value_transposed));

    const auto sdp_reshape_dims =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{batch_size, seq_size, neg_one_1d}, 0));
    const auto sdp_transpose_dims =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{zero_1d, two_1d, one_1d, three_1d}, 0));

    const auto scaled_dot_product_attention_transposed =
        context.mark_node(std::make_shared<opset10::Transpose>(scaled_dot_product_attention, sdp_transpose_dims));
    const auto scaled_dot_product_attention_reshaped = context.mark_node(
        std::make_shared<opset10::Reshape>(scaled_dot_product_attention_transposed, sdp_reshape_dims, false));

    const auto scaled_dot_product_attention_weighted =
        context.mark_node(std::make_shared<opset10::MatMul>(scaled_dot_product_attention_reshaped, proj_weight));
    const auto scaled_dot_product_attention_biased =
        context.mark_node(std::make_shared<opset10::Add>(scaled_dot_product_attention_weighted, proj_bias));

    return {scaled_dot_product_attention_biased, scaled_dot_product_attention_biased};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
