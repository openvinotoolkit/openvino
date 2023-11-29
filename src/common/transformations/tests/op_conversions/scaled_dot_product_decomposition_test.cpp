// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
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
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

using namespace ov;
using namespace testing;

const std::shared_ptr<ov::Node> scaled_dot_product_attention_decomposition(
    const std::shared_ptr<ov::Node> query,
    const std::shared_ptr<ov::Node> key,
    const std::shared_ptr<ov::Node> value,
    const std::shared_ptr<ov::Node> attention_mask,
    const std::shared_ptr<ov::Node> scale,
    const bool casual);

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecompositionStatic) {
    const PartialShape query_key_value_mask_shape{1, 32, 32};
    const PartialShape scale_shape{};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    const auto casual = false;
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);

        model = std::make_shared<ov::Model>(NodeVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask, scale});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecompositionDynamic) {
    const PartialShape query_key_value_mask_shape{-1, -1, -1};
    const PartialShape scale_shape{};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, query_key_value_mask_shape);
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    const auto casual = false;
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);

        model = std::make_shared<ov::Model>(NodeVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask, scale});
    }
}

const std::shared_ptr<ov::Node> scaled_dot_product_attention_decomposition(
    const std::shared_ptr<ov::Node> query,
    const std::shared_ptr<ov::Node> key,
    const std::shared_ptr<ov::Node> value,
    const std::shared_ptr<ov::Node> attention_mask,
    const std::shared_ptr<ov::Node> scale,
    const bool casual) {
    const auto q_shape = std::make_shared<ov::op::v3::ShapeOf>(query, element::i32);
    const auto k_shape = std::make_shared<ov::op::v3::ShapeOf>(key, element::i32);
    const auto minus_one = ov::op::v0::Constant::create(element::i32, Shape{}, {-1});
    const auto minus_two = ov::op::v0::Constant::create(element::i32, Shape{}, {-2});
    const auto zero_i = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    const auto one_i = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    const auto one_f = std::make_shared<ov::op::v1::ConvertLike>(one_i, query);
    const auto zero_f = std::make_shared<ov::op::v1::ConvertLike>(zero_i, query);

    const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale);
    auto k_rank = std::make_shared<ov::op::v3::ShapeOf>(k_shape, element::i32)->output(0);
    const auto k_last_dim = std::make_shared<ov::op::v1::Add>(k_rank, minus_one);
    const auto k_next_dim = std::make_shared<ov::op::v1::Add>(k_rank, minus_two)->output(0);
    k_rank = std::make_shared<ov::op::v0::Squeeze>(k_rank, zero_i);
    auto minus_inf =
        ov::op::v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()})->output(0);
    const auto keep_dim_last = std::make_shared<ov::op::v0::Squeeze>(k_next_dim, zero_i);
    const auto k_dims_before_transpose =
        std::make_shared<ov::op::v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

    const auto transpose_dims =
        std::make_shared<ov::op::v0::Concat>(OutputVector{k_dims_before_transpose, k_last_dim, k_next_dim}, 0);
    const auto k_transposed = std::make_shared<ov::op::v1::Transpose>(key, transpose_dims);
    auto scaled_atten = std::make_shared<ov::op::v0::MatMul>(q_scaled, k_transposed)->output(0);
    minus_inf = std::make_shared<ov::op::v1::ConvertLike>(minus_inf, scaled_atten);

    Output<Node> mask;
    Output<Node> atten_mask;
    if (!casual) {
        mask = attention_mask;
        if (mask.get_element_type() == element::boolean) {
            atten_mask = std::make_shared<ov::op::v1::ConvertLike>(mask, scaled_atten);
            const auto inv_mask = std::make_shared<ov::op::v1::LogicalNot>(mask);
            atten_mask = std::make_shared<ov::op::v1::Select>(inv_mask, atten_mask, minus_inf);
        } else {
            atten_mask = mask;
        }
    } else {
        const auto target_s_len = std::make_shared<ov::op::v8::Gather>(q_shape, minus_two, zero_i);
        const auto source_s_len = std::make_shared<ov::op::v8::Gather>(k_shape, minus_two, zero_i);
        const auto ssl = std::make_shared<ov::op::v0::Unsqueeze>(source_s_len, zero_i);
        const auto tsl = std::make_shared<ov::op::v0::Unsqueeze>(target_s_len, zero_i);
        const auto mask_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{tsl, ssl}, 0);
        mask = std::make_shared<ov::op::v1::Broadcast>(minus_inf, mask_shape);
        auto horizontal_range =
            std::make_shared<ov::op::v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
        horizontal_range = std::make_shared<ov::op::v0::Unsqueeze>(horizontal_range, zero_i);
        const auto stop = std::make_shared<ov::op::v1::Add>(target_s_len, one_i);
        auto vertical_range = std::make_shared<ov::op::v4::Range>(one_i, stop, one_i, element::i32)->output(0);
        vertical_range = std::make_shared<ov::op::v0::Unsqueeze>(vertical_range, one_i);
        const auto triu = std::make_shared<ov::op::v1::GreaterEqual>(horizontal_range, vertical_range);
        atten_mask = std::make_shared<ov::op::v1::Select>(triu, mask, zero_f);
    }
    scaled_atten = std::make_shared<ov::op::v1::Add>(scaled_atten, atten_mask);

    scaled_atten = std::make_shared<ov::op::v8::Softmax>(scaled_atten, -1);
    const std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::MatMul>(scaled_atten, value);
    return result;
}
