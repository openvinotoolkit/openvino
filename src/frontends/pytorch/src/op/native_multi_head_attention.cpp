// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_native_multi_head_attention(const NodeContext& context) {
    /*
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
    const auto embed_dim = context.get_input(3);
    const auto num_head = context.get_input(4);
    const auto qkv_weight = context.get_input(5);
    const auto qkv_bias = context.get_input(6);
    const auto proj_weight = context.get_input(7);
    const auto proj_bias = context.get_input(8);
    const auto need_weights = context.const_input<bool>(10);
    const auto average_weights = context.const_input<bool>(11);

    const auto minus_inf =
        context.mark_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}));
    const auto embed_dim_i64 = context.mark_node(std::make_shared<v0::Convert>(embed_dim, element::i64));
    const auto num_head_i64 = context.mark_node(std::make_shared<v0::Convert>(num_head, element::i64));

    const auto neg_one_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    const auto zero_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    const auto one_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    const auto two_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {2}));
    const auto three_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {3}));
    const auto heads_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(num_head_i64, zero_1d));

    const auto ev_1_slice_1d = context.mark_node(std::make_shared<v1::Multiply>(one_1d, embed_dim_i64));
    const auto ev_2_slice_1d = context.mark_node(std::make_shared<v1::Multiply>(two_1d, embed_dim_i64));
    const auto ev_3_slice_1d = context.mark_node(std::make_shared<v1::Multiply>(three_1d, embed_dim_i64));

    const auto qkv_shape = context.mark_node(std::make_shared<v3::ShapeOf>(query));
    const auto batch_size = context.mark_node(std::make_shared<v8::Gather>(qkv_shape, zero_1d, zero_1d));
    const auto seq_size = context.mark_node(std::make_shared<v8::Gather>(qkv_shape, one_1d, zero_1d));
    const auto embed_div_heads = context.mark_node(std::make_shared<v1::Divide>(embed_dim_i64, heads_1d, true));

    const auto query_proj_weight =
        context.mark_node(std::make_shared<v8::Slice>(qkv_weight, zero_1d, ev_1_slice_1d, one_1d, zero_1d));
    const auto key_proj_weight =
        context.mark_node(std::make_shared<v8::Slice>(qkv_weight, ev_1_slice_1d, ev_2_slice_1d, one_1d, zero_1d));
    const auto value_proj_weight =
        context.mark_node(std::make_shared<v8::Slice>(qkv_weight, ev_2_slice_1d, ev_3_slice_1d, one_1d, zero_1d));
    const auto query_proj_bias =
        context.mark_node(std::make_shared<v8::Slice>(qkv_bias, zero_1d, ev_1_slice_1d, one_1d, zero_1d));
    const auto key_proj_bias =
        context.mark_node(std::make_shared<v8::Slice>(qkv_bias, ev_1_slice_1d, ev_2_slice_1d, one_1d, zero_1d));
    const auto value_proj_bias =
        context.mark_node(std::make_shared<v8::Slice>(qkv_bias, ev_2_slice_1d, ev_3_slice_1d, one_1d, zero_1d));

    const auto query_weighted = context.mark_node(std::make_shared<v0::MatMul>(query, query_proj_weight, false, true));
    const auto key_weighted = context.mark_node(std::make_shared<v0::MatMul>(key, key_proj_weight, false, true));
    const auto value_weighted = context.mark_node(std::make_shared<v0::MatMul>(value, value_proj_weight, false, true));

    const auto query_biased = context.mark_node(std::make_shared<v1::Add>(query_weighted, query_proj_bias));
    const auto key_biased = context.mark_node(std::make_shared<v1::Add>(key_weighted, key_proj_bias));
    const auto value_biased = context.mark_node(std::make_shared<v1::Add>(value_weighted, value_proj_bias));

    const auto qkv_reshape_dims =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_size, seq_size, heads_1d, neg_one_1d}, 0));
    const auto qv_transpose_dims = context.mark_node(v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    const auto k_transpose_dims = context.mark_node(v0::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1}));

    const auto query_reshaped = context.mark_node(std::make_shared<v1::Reshape>(query_biased, qkv_reshape_dims, false));
    const auto key_reshaped = context.mark_node(std::make_shared<v1::Reshape>(key_biased, qkv_reshape_dims, false));
    const auto value_reshaped = context.mark_node(std::make_shared<v1::Reshape>(value_biased, qkv_reshape_dims, false));

    const auto query_transposed = context.mark_node(std::make_shared<v1::Transpose>(query_reshaped, qv_transpose_dims));
    const auto key_transposed = context.mark_node(std::make_shared<v1::Transpose>(key_reshaped, k_transpose_dims));
    const auto value_transposed = context.mark_node(std::make_shared<v1::Transpose>(value_reshaped, qv_transpose_dims));

    const auto scale_one = context.mark_node(std::make_shared<v1::ConvertLike>(one_1d, query_transposed));
    const auto scale_dim = context.mark_node(std::make_shared<v1::ConvertLike>(embed_div_heads, query_transposed));
    const auto scale_dim_sqrt = context.mark_node(std::make_shared<v0::Sqrt>(scale_dim));
    const auto scale = context.mark_node(std::make_shared<v1::Divide>(scale_one, scale_dim_sqrt));

    const auto query_key_transpose_dot_product =
        context.mark_node(std::make_shared<v0::MatMul>(query_transposed, key_transposed));

    auto scaled_dot_product = context.mark_node(std::make_shared<v1::Multiply>(query_key_transpose_dot_product, scale));

    // Mask handling
    if (!context.input_is_none(9) && !context.input_is_none(12)) {
        auto atten_mask = context.get_input(9);
        // Only allow boolean masks - PyTorch automatically converts
        // non-boolean to boolean masks
        if (atten_mask.get_element_type() == element::boolean) {
            const auto minus_inf_conv =
                context.mark_node(std::make_shared<v1::ConvertLike>(minus_inf, scaled_dot_product));
            const auto mask_inverse = context.mark_node(std::make_shared<v1::LogicalNot>(atten_mask));
            atten_mask = context.mark_node(std::make_shared<v1::ConvertLike>(atten_mask, scaled_dot_product));
            atten_mask = context.mark_node(std::make_shared<v1::Select>(mask_inverse, atten_mask, minus_inf_conv));
        } else {
            // Once int/float mask type is supported in PyTorch,
            // remove this assert to allow for such masks in OV
            PYTORCH_OP_CONVERSION_CHECK(1, "Non-boolean masks are not supported.");
            atten_mask = context.mark_node(std::make_shared<v1::ConvertLike>(atten_mask, scaled_dot_product));
        }

        // If mask type is 1 (only key-padding) then mask's shape is (N, S).
        // It must be reshaped to (N, 1, 1, S) to properly broadcast it proper dims in the next step
        if (context.const_input<int64_t>(12) == 1) {
            const auto target_mask_reshape =
                context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_size, one_1d, one_1d, seq_size}, 0));
            atten_mask = context.mark_node(std::make_shared<v1::Reshape>(atten_mask, target_mask_reshape, false));
        }

        // All mask types should be broadcast to this shape,
        // Except for type 2 which already has this shape
        if (context.const_input<int64_t>(12) != 2) {
            const auto target_mask_shape = context.mark_node(
                std::make_shared<v0::Concat>(OutputVector{batch_size, heads_1d, seq_size, seq_size}, 0));
            atten_mask = context.mark_node(std::make_shared<v3::Broadcast>(atten_mask, target_mask_shape));
        }
        scaled_dot_product = context.mark_node(std::make_shared<v1::Add>(scaled_dot_product, atten_mask));
    }

    const auto scaled_dot_product_softmax = context.mark_node(std::make_shared<v8::Softmax>(scaled_dot_product, -1));
    const auto scaled_dot_product_attention =
        context.mark_node(std::make_shared<v0::MatMul>(scaled_dot_product_softmax, value_transposed));

    const auto sdp_reshape_dims =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_size, seq_size, neg_one_1d}, 0));
    // Undo transpose (transpose back to original qv shape)
    const auto scaled_dot_product_attention_transposed =
        context.mark_node(std::make_shared<v1::Transpose>(scaled_dot_product_attention, qv_transpose_dims));
    const auto scaled_dot_product_attention_reshaped = context.mark_node(
        std::make_shared<v1::Reshape>(scaled_dot_product_attention_transposed, sdp_reshape_dims, false));

    const auto scaled_dot_product_attention_weighted = context.mark_node(
        std::make_shared<v0::MatMul>(scaled_dot_product_attention_reshaped, proj_weight, false, true));
    const auto scaled_dot_product_attention_biased =
        context.mark_node(std::make_shared<v1::Add>(scaled_dot_product_attention_weighted, proj_bias));

    if (average_weights) {
        const auto target_div_shape = context.mark_node(std::make_shared<v3::ShapeOf>(scaled_dot_product));
        const auto heads_div = context.mark_node(std::make_shared<v3::Broadcast>(heads_1d, target_div_shape));
        const auto heads_div_conv = context.mark_node(std::make_shared<v1::ConvertLike>(heads_div, scaled_dot_product));
        scaled_dot_product = context.mark_node(std::make_shared<v1::Divide>(scaled_dot_product, heads_div_conv, false));
        scaled_dot_product = context.mark_node(std::make_shared<v1::ReduceSum>(scaled_dot_product, one_1d));
    }

    if (need_weights) {
        return {scaled_dot_product_attention_biased, scaled_dot_product};
    } else {
        // When need_weights == false, returns None as a second output
        const auto none = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs());
        auto attrs = none->get_attrs();
        attrs["none_value"] = "";
        none->set_attrs(attrs);
        const auto none_marked = context.mark_node(none);
        return {scaled_dot_product_attention_biased, none_marked};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
