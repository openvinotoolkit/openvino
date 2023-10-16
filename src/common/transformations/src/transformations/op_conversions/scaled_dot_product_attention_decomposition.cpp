// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
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
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ScaledDotProductAttentionDecomposition::ScaledDotProductAttentionDecomposition() {
    MATCHER_SCOPE(ScaledDotProductAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(
            pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }

        auto new_output_node = decompose(node)[0].get_node_shared_ptr();
        new_output_node->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, new_output_node);
        ov::replace_node(node, new_output_node);
        std::cerr << "[ DEBUG ] ScaledDotProductAttention was decomposed in compile_model\n";
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

ov::OutputVector ov::pass::ScaledDotProductAttentionDecomposition::decompose(
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node) const {
    using namespace ov::op;
    auto query = node->input_value(0);
    auto key = node->input_value(1);
    auto value = node->input_value(2);
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
    auto minus_inf = v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()})->output(0);
    auto keep_dim_last = std::make_shared<v0::Squeeze>(k_next_dim, zero_i);
    auto k_dims_before_transpose = std::make_shared<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

    auto transpose_dims =
        std::make_shared<v0::Concat>(OutputVector{k_dims_before_transpose, k_last_dim, k_next_dim}, 0);
    auto k_transposed = std::make_shared<v1::Transpose>(key, transpose_dims);
    auto scaled_atten = std::make_shared<v0::MatMul>(q_scaled, k_transposed)->output(0);
    minus_inf = std::make_shared<v1::ConvertLike>(minus_inf, scaled_atten);
    // two types of masks are supported. A boolean mask where a value of True indicates that the element should take
    // part in attention. A float mask of the same type as query, key, value that is added to the attention score.
    if (node->get_causal() || node->get_input_size() > 3) {
        Output<Node> mask;
        Output<Node> atten_mask;
        if (node->get_input_size() > 3) {
            mask = node->input_value(3);
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
            auto horizontal_range = std::make_shared<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
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
