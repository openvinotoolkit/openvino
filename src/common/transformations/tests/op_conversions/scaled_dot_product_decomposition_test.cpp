// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

using namespace ov;
using namespace testing;

const std::shared_ptr<ov::Node> scaled_dot_product_attention_decomposition(std::shared_ptr<ov::Node> query,
                                                                           std::shared_ptr<ov::Node> key,
                                                                           std::shared_ptr<ov::Node> value,
                                                                           std::shared_ptr<ov::Node> attention_mask,
                                                                           std::shared_ptr<ov::Node> scale,
                                                                           bool casual,
                                                                           bool scale_after_matmul = false,
                                                                           std::shared_ptr<ov::Node> sinks = nullptr);

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecompositionStaticBasic) {
    const PartialShape query_shape{1, 32, 32};
    const PartialShape key_shape{1, 32, 32};
    const PartialShape value_shape{1, 32, 32};
    const PartialShape attention_mask_shape{1, 32, 32};
    const PartialShape scale_shape{1};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    const auto casual = false;
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);

        model = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask, scale});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecompositionStaticBroadcastKey) {
    const PartialShape query_shape{2, 1, 3, 1};
    const PartialShape key_shape{1, 7, 4, 1};
    const PartialShape value_shape{2, 7, 4, 1};
    const PartialShape attention_mask_shape{1, 1, 3, 4};
    const PartialShape scale_shape{1};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    const auto casual = false;
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);

        model = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask, scale});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecompositionStaticBroadcastQuery) {
    const PartialShape query_shape{1, 7, 3, 4};
    const PartialShape key_shape{2, 1, 1, 4};
    const PartialShape value_shape{1, 1, 1, 6};
    const PartialShape attention_mask_shape{1, 1, 3, 1};
    const PartialShape scale_shape{};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    const auto casual = false;
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);

        model = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual, true);
        model_ref = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask, scale});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionCasualPartiallyDynamic) {
    const PartialShape query_shape{-1, -1, 24, 64};
    const PartialShape key_shape{-1, -1, 24, 64};
    const PartialShape value_shape{-1, -1, -1, 64};
    const PartialShape attention_mask_shape{-1, -1, -1, -1};
    const auto casual = true;

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, casual);

        model = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, nullptr, casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecompositionDynamic) {
    const PartialShape query_shape{-1, -1, -1};
    const PartialShape key_shape{-1, -1, -1};
    const PartialShape value_shape{-1, -1, -1};
    const PartialShape attention_mask_shape{-1, -1, -1};
    const PartialShape scale_shape{};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    const auto casual = false;
    {
        const auto scaled_dot_product_attention =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);

        model = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                            ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        const auto scaled_dot_product_attention =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{scaled_dot_product_attention},
                                                ParameterVector{query, key, value, attention_mask, scale});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecomposition_ScalarScale_MultiplyAfterMatMul) {
    const PartialShape query_shape{1, 32, 64};
    const PartialShape key_shape{1, 32, 64};
    const PartialShape value_shape{1, 32, 64};
    const PartialShape attention_mask_shape{1, 32, 32};

    // Constant scalar scale
    auto scale = ov::op::v0::Constant::create(element::f32, Shape{}, {0.125f});

    auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);

    const bool casual = false;

    {
        auto sdp =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);
        model = std::make_shared<ov::Model>(OutputVector{sdp}, ParameterVector{query, key, value, attention_mask});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        auto ref = scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual, true);
        model_ref = std::make_shared<ov::Model>(OutputVector{ref}, ParameterVector{query, key, value, attention_mask});
    }
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecomposition_DynamicScale_MultiplyBeforeMatMul) {
    const PartialShape query_shape{-1, -1, 64};
    const PartialShape key_shape{-1, -1, 64};
    const PartialShape value_shape{-1, -1, 64};
    const PartialShape attention_mask_shape{-1, -1, -1};
    const PartialShape scale_shape{1};

    auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, scale_shape);
    auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, attention_mask_shape);

    const bool casual = false;

    {
        auto sdp =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, casual);
        model =
            std::make_shared<ov::Model>(OutputVector{sdp}, ParameterVector{query, key, value, attention_mask, scale});
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    {
        auto ref = scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual, false);
        model_ref =
            std::make_shared<ov::Model>(OutputVector{ref}, ParameterVector{query, key, value, attention_mask, scale});
    }
}

const std::shared_ptr<ov::Node> scaled_dot_product_attention_decomposition(std::shared_ptr<ov::Node> query,
                                                                           std::shared_ptr<ov::Node> key,
                                                                           std::shared_ptr<ov::Node> value,
                                                                           std::shared_ptr<ov::Node> attention_mask,
                                                                           std::shared_ptr<ov::Node> scale,
                                                                           bool casual,
                                                                           bool scale_after_matmul,
                                                                           std::shared_ptr<ov::Node> sinks) {
    const auto q_shape = std::make_shared<ov::op::v3::ShapeOf>(query, element::i32);
    const auto k_shape = std::make_shared<ov::op::v3::ShapeOf>(key, element::i32);
    const auto minus_one = ov::op::v0::Constant::create(element::i32, Shape{}, {-1});
    const auto minus_two = ov::op::v0::Constant::create(element::i32, Shape{}, {-2});
    const auto zero_i = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    const auto one_i = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    const auto one_f = std::make_shared<ov::op::v1::ConvertLike>(one_i, query);
    const auto zero_f = std::make_shared<ov::op::v1::ConvertLike>(zero_i, query);

    auto extract_dim = [&zero_i](const std::shared_ptr<ov::op::v3::ShapeOf>& shape_of,
                                 const int64_t idx) -> std::shared_ptr<ov::Node> {
        const auto& shape = shape_of->get_input_partial_shape(0);
        const auto& dim = shape[idx];
        if (dim.is_static()) {
            return ov::op::v0::Constant::create(element::i32, Shape{}, {dim.get_length()});
        }
        const auto dim_to_extract_const = ov::op::v0::Constant::create(element::i32, Shape{}, {idx});
        return std::make_shared<ov::op::v8::Gather>(shape_of, dim_to_extract_const, zero_i);
    };

    if (scale == nullptr) {
        scale = extract_dim(q_shape, -1);
        scale = std::make_shared<ov::op::v1::ConvertLike>(scale, query);
        auto sqrt_scale = std::make_shared<ov::op::v0::Sqrt>(scale);
        scale = std::make_shared<ov::op::v1::Divide>(one_f, sqrt_scale);
    }
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

    Output<Node> scaled_atten;
    if (scale_after_matmul) {
        const auto atten = std::make_shared<ov::op::v0::MatMul>(query, k_transposed)->output(0);
        scaled_atten = std::make_shared<ov::op::v1::Multiply>(atten, scale);
    } else {
        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale);
        scaled_atten = std::make_shared<ov::op::v0::MatMul>(q_scaled, k_transposed)->output(0);
    }
    minus_inf = std::make_shared<ov::op::v1::ConvertLike>(minus_inf, scaled_atten);

    Output<Node> mask;
    Output<Node> atten_mask;
    if (!casual) {
        mask = attention_mask;
        if (mask.get_element_type() == element::boolean) {
            atten_mask = std::make_shared<ov::op::v1::Select>(mask, zero_f, minus_inf);
        } else {
            atten_mask = mask;
        }
    } else {
        const auto target_s_len = extract_dim(q_shape, -2);
        const auto source_s_len = extract_dim(k_shape, -2);
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

    if (sinks) {
        auto minus_two = ov::op::v0::Constant::create(element::i32, Shape{1}, {-2});
        auto minus_one = ov::op::v0::Constant::create(element::i32, Shape{1}, {-1});
        auto zero_i = ov::op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto one_i = ov::op::v0::Constant::create(element::i32, Shape{1}, {1});

        auto q_last_but_one_dim =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v0::ShapeOf>(q_shape),
                                                   ov::op::v0::Constant::create(ov::element::i64, Shape{}, {1}));
        auto sinks_target_shape_1 = std::make_shared<ov::op::v8::Slice>(q_shape, zero_i, q_last_but_one_dim, one_i);
        auto sinks_target_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{sinks_target_shape_1, one_i}, 0);
        auto sinks_broadcast = std::make_shared<ov::op::v1::Broadcast>(sinks, sinks_target_shape);

        auto scaled_attn_sink = std::make_shared<ov::op::v0::Concat>(OutputVector{scaled_atten, sinks_broadcast}, -1);
        scaled_atten = std::make_shared<ov::op::v8::Softmax>(scaled_attn_sink, -1);

        auto seq_len = std::make_shared<ov::op::v8::Gather>(q_shape, minus_two, zero_i);
        scaled_atten = std::make_shared<ov::op::v8::Slice>(scaled_atten, zero_i, seq_len, one_i, minus_one);
    } else {
        scaled_atten = std::make_shared<ov::op::v8::Softmax>(scaled_atten, -1);
    }

    const std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::MatMul>(scaled_atten, value);
    return result;
}

TEST_F(TransformationTestsF, ScaledDotProductAttentionDecomposition_Sinks) {
    const PartialShape query_shape{-1, 64, -1, 64};
    const PartialShape key_shape{-1, 64, -1, 64};
    const PartialShape value_shape{-1, 64, -1, 64};
    const PartialShape attention_mask_shape{-1, 1, -1, -1};
    const PartialShape scale_shape{1};
    const PartialShape sinks_shape{1, 64, 1, 1};

    auto scale = std::make_shared<ov::op::v0::Parameter>(element::f16, scale_shape);
    auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    auto sinks = std::make_shared<ov::op::v0::Parameter>(element::f16, sinks_shape);

    const bool casual = false;

    {
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                             key,
                                                                             value,
                                                                             attention_mask,
                                                                             scale,
                                                                             sinks,
                                                                             casual);
        model = std::make_shared<ov::Model>(OutputVector{sdpa},
                                            ParameterVector{query, key, value, attention_mask, scale, sinks});
        manager.register_pass<ov::pass::VisualizeTree>("before.svg");
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.register_pass<ov::pass::VisualizeTree>("after.svg");
    }

    {
        auto ref =
            scaled_dot_product_attention_decomposition(query, key, value, attention_mask, scale, casual, false, sinks);
        model_ref = std::make_shared<ov::Model>(OutputVector{ref},
                                                ParameterVector{query, key, value, attention_mask, scale, sinks});
        ov::pass::VisualizeTree("model_ref.svg").run_on_model(model_ref);
    }
}