// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

std::shared_ptr<ov::Node> translate_scaled_dot_product_attention_common(const NodeContext& context) {
    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);
    auto q_shape = context.mark_node(std::make_shared<v3::ShapeOf>(query, element::i32));
    auto k_shape = context.mark_node(std::make_shared<v3::ShapeOf>(key, element::i32));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto minus_two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-2}));
    auto zero_i = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one_i = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto scale = context.mark_node(std::make_shared<v8::Gather>(q_shape, minus_one, zero_i));
    scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, query));
    auto sqrt_scale = context.mark_node(std::make_shared<v0::Sqrt>(scale));
    auto one_f = context.mark_node(std::make_shared<v1::ConvertLike>(one_i, sqrt_scale));
    auto zero_f = context.mark_node(std::make_shared<v1::ConvertLike>(zero_i, sqrt_scale));
    scale = context.mark_node(std::make_shared<v1::Divide>(one_f, sqrt_scale));
    auto q_scaled = context.mark_node(std::make_shared<v1::Multiply>(query, scale));
    auto k_rank = context.mark_node(std::make_shared<v3::ShapeOf>(k_shape, element::i32));
    auto k_last_dim = context.mark_node(std::make_shared<v1::Add>(k_rank, minus_one));
    auto k_next_dim = context.mark_node(std::make_shared<v1::Add>(k_rank, minus_two));
    k_rank = context.mark_node(std::make_shared<v0::Squeeze>(k_rank, zero_i));
    auto minus_inf =
        context.mark_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}));
    auto keep_dim_last = context.mark_node(std::make_shared<v0::Squeeze>(k_next_dim, zero_i));
    auto k_dims_before_transpose =
        context.mark_node(std::make_shared<v4::Range>(zero_i, keep_dim_last, one_i, element::i32));

    auto transpose_dims = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{k_dims_before_transpose, k_last_dim, k_next_dim}, 0));
    auto k_transposed = context.mark_node(std::make_shared<v1::Transpose>(key, transpose_dims));
    auto scaled_atten = context.mark_node(std::make_shared<v0::MatMul>(q_scaled, k_transposed));
    minus_inf = context.mark_node(std::make_shared<v1::ConvertLike>(minus_inf, scaled_atten));
    // two types of masks are supported. A boolean mask where a value of True indicates that the element should take
    // part in attention. A float mask of the same type as query, key, value that is added to the attention score.
    auto is_causal = false;
    if (!context.input_is_none(5)) {
        is_causal = context.const_input<bool>(5);
    }
    if (is_causal || !context.input_is_none(3)) {
        Output<Node> mask;
        Output<Node> atten_mask;
        if (!context.input_is_none(3)) {
            mask = context.get_input(3);
            if (mask.get_element_type() == element::boolean) {
                atten_mask = context.mark_node(std::make_shared<v1::ConvertLike>(mask, scaled_atten));
                auto inv_mask = context.mark_node(std::make_shared<v1::LogicalNot>(mask));
                atten_mask = context.mark_node(std::make_shared<v1::Select>(inv_mask, atten_mask, minus_inf));
            } else {
                atten_mask = mask;
            }
        } else {
            auto target_s_len = context.mark_node(std::make_shared<v8::Gather>(q_shape, minus_two, zero_i));
            auto source_s_len = context.mark_node(std::make_shared<v8::Gather>(k_shape, minus_two, zero_i));
            auto ssl = context.mark_node(std::make_shared<v0::Unsqueeze>(source_s_len, zero_i));
            auto tsl = context.mark_node(std::make_shared<v0::Unsqueeze>(target_s_len, zero_i));
            auto mask_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{tsl, ssl}, 0));
            mask = context.mark_node(std::make_shared<v1::Broadcast>(minus_inf, mask_shape));
            auto horizontal_range =
                context.mark_node(std::make_shared<v4::Range>(zero_i, source_s_len, one_i, element::i32));
            horizontal_range = context.mark_node(std::make_shared<v0::Unsqueeze>(horizontal_range, zero_i));
            auto stop = context.mark_node(std::make_shared<v1::Add>(target_s_len, one_i));
            auto vertical_range = context.mark_node(std::make_shared<v4::Range>(one_i, stop, one_i, element::i32));
            vertical_range = context.mark_node(std::make_shared<v0::Unsqueeze>(vertical_range, one_i));
            auto triu = context.mark_node(std::make_shared<v1::GreaterEqual>(horizontal_range, vertical_range));
            atten_mask = context.mark_node(std::make_shared<v1::Select>(triu, mask, zero_f));
        }
        scaled_atten = context.mark_node(std::make_shared<v1::Add>(scaled_atten, atten_mask));
    }
    scaled_atten = context.mark_node(std::make_shared<v8::Softmax>(scaled_atten, -1));
    return context.mark_node(std::make_shared<v0::MatMul>(scaled_atten, value));
};

OutputVector translate_scaled_dot_product_attention(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False)
    num_inputs_check(context, 6, 6);
    return {translate_scaled_dot_product_attention_common(context)};
};

OutputVector translate_scaled_dot_product_attention_fx(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False)
    num_inputs_check(context, 3, 6);
    auto output = translate_scaled_dot_product_attention_common(context);
    // TODO: scaled_dot_product_flash_attention has 9 outputs but for most cases only
    // the first input is used. Rest of the outputs should be returned properly as
    // needed.
    ov::OutputVector out_vec;
    out_vec.push_back(output);
    return {context.mark_node(make_list_construct(out_vec))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
