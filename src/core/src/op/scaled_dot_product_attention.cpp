// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"

#include "itt.hpp"

using namespace std;
using namespace ov;

op::v12::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
            const Output<Node>& key,
            const Output<Node>& value,
            bool is_causal,
            const Output<Node>& attn_mask)
    : op::Op(attn_mask.get_node_shared_ptr() ? OutputVector{query, key, value, attn_mask} : OutputVector{query, key, value}),
    m_is_causal(is_causal) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v12::ScaledDotProductAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ScaledDotProductAttention_clone_with_new_inputs);
    return make_shared<v12::ScaledDotProductAttention>(new_args[0], new_args[1], new_args[2], m_is_causal, new_args.size() == 4 ? new_args[3] : Output<Node>());
}

void op::v12::ScaledDotProductAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ScaledDotProductAttention_validate_and_infer_types);
    auto outputs = decompose();
    NODE_VALIDATION_CHECK(this, outputs.size() == 1);
    auto output = outputs[0];
    set_output_type(0, output.get_element_type(), output.get_partial_shape());
}

OutputVector op::v12::ScaledDotProductAttention::decompose() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 3 || get_input_size() == 4);
    auto query = input_value(0);
    auto key = input_value(1);
    auto value = input_value(2);
    auto q_shape = std::make_shared<v3::ShapeOf>(query, element::i32);
    auto k_shape = std::make_shared<v3::ShapeOf>(key, element::i32);
    auto minus_one = v0::Constant::create(element::i32, Shape{}, {-1});
    auto minus_two = v0::Constant::create(element::i32, Shape{}, {-2});
    auto zero_i = v0::Constant::create(element::i32, Shape{}, {0});
    auto one_i = v0::Constant::create(element::i32, Shape{}, {1});
    auto scale = std::make_shared<v8::Gather>(q_shape, minus_one, zero_i)->output(0);
    scale = std::make_shared<v1::ConvertLike>(scale, query);
    auto sqrt_scale = std::make_shared<v0::Sqrt>(scale);
    auto one_f = std::make_shared<v1::ConvertLike>(one_i, sqrt_scale);
    auto zero_f = std::make_shared<v1::ConvertLike>(zero_i, sqrt_scale);
    scale = std::make_shared<v1::Divide>(one_f, sqrt_scale);
    auto q_scaled = std::make_shared<v1::Multiply>(query, scale);
    auto k_rank = std::make_shared<v3::ShapeOf>(k_shape, element::i32)->output(0);
    auto k_last_dim = std::make_shared<v1::Add>(k_rank, minus_one);
    auto k_next_dim = std::make_shared<v1::Add>(k_rank, minus_two)->output(0);
    k_rank = std::make_shared<v0::Squeeze>(k_rank, zero_i);
    auto minus_inf =
        v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()})->output(0);
    auto keep_dim_last = std::make_shared<v0::Squeeze>(k_next_dim, zero_i);
    auto k_dims_before_transpose =
        std::make_shared<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

    auto transpose_dims =
        std::make_shared<v0::Concat>(OutputVector{k_dims_before_transpose, k_last_dim, k_next_dim}, 0);
    auto k_transposed = std::make_shared<v1::Transpose>(key, transpose_dims);
    auto scaled_atten = std::make_shared<v0::MatMul>(q_scaled, k_transposed)->output(0);
    minus_inf = std::make_shared<v1::ConvertLike>(minus_inf, scaled_atten);
    // two types of masks are supported. A boolean mask where a value of True indicates that the element should take
    // part in attention. A float mask of the same type as query, key, value that is added to the attention score.
    if (m_is_causal || get_input_size() > 3) {
        Output<Node> mask;
        Output<Node> atten_mask;
        if (get_input_size() > 3) {
            mask = input_value(3);
            if (mask.get_element_type() == element::boolean) {
                atten_mask = std::make_shared<v1::ConvertLike>(mask, scaled_atten);
                auto inv_mask = std::make_shared<v1::LogicalNot>(mask);
                atten_mask = std::make_shared<v1::Select>(inv_mask, atten_mask, minus_inf);
            } else {
                atten_mask = mask;
            }
        } else {
            auto target_s_len = std::make_shared<v8::Gather>(q_shape, minus_two, zero_i);
            auto source_s_len = std::make_shared<v8::Gather>(k_shape, minus_two, zero_i);
            auto ssl = std::make_shared<v0::Unsqueeze>(source_s_len, zero_i);
            auto tsl = std::make_shared<v0::Unsqueeze>(target_s_len, zero_i);
            auto mask_shape = std::make_shared<v0::Concat>(OutputVector{tsl, ssl}, 0);
            mask = std::make_shared<v1::Broadcast>(minus_inf, mask_shape);
            auto horizontal_range =
                std::make_shared<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
            horizontal_range = std::make_shared<v0::Unsqueeze>(horizontal_range, zero_i);
            auto stop = std::make_shared<v1::Add>(target_s_len, one_i);
            auto vertical_range = std::make_shared<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
            vertical_range = std::make_shared<v0::Unsqueeze>(vertical_range, one_i);
            auto triu = std::make_shared<v1::GreaterEqual>(horizontal_range, vertical_range);
            atten_mask = std::make_shared<v1::Select>(triu, mask, zero_f);
        }
        scaled_atten = std::make_shared<v1::Add>(scaled_atten, atten_mask);
    }
    scaled_atten = std::make_shared<v8::Softmax>(scaled_atten, -1);
    return {std::make_shared<v0::MatMul>(scaled_atten, value)};
}

bool op::v12::ScaledDotProductAttention::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("causal", m_is_causal);
    return true;
}