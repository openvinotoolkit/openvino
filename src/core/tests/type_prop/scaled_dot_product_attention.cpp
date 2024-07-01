// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace testing;

TEST(type_prop, scaled_dot_product_attention_static_5_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, Shape{1, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, Shape{1});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (Shape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_4_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 3, 5});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_3_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_3_inputs_causal) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    auto causal = true;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_scalar_attention_causal_false) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_ignored_attention_mask) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 8, 9, 10, 11});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = true;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_5_inputs_extra_batch) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 1, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_4_inputs_extra_batch) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 1, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 1, 3, 5});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_3_inputs_extra_batch) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 6});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_3_inputs_extra_batch_causal_true) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 5, 6});
    auto causal = true;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_ignored_attention_mask_extra_batch) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 1, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 7, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 8, 9, 10, 11});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = true;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_5_inputs_dynamic_rank) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, scaled_dot_product_attention_dynamic_3d) {
    const auto dynamic = PartialShape{-1, -1, -1};
    const auto query = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto key = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto value = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (dynamic));
}

TEST(type_prop, scaled_dot_product_attention_dynamic_4d) {
    const auto dynamic = PartialShape{-1, -1, -1, -1};
    const auto query = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto key = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto value = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (dynamic));
}

TEST(type_prop, scaled_dot_product_attention_mixed_shape_infer_5_inputs) {
    PartialShape query_shape{{2, 4}, 3, {2, 5}, 4};
    auto symbols = set_shape_symbols(query_shape);
    PartialShape key_shape{{4, 8}, {2, 4}, 5, 4};
    set_shape_symbols(key_shape);
    PartialShape value_shape{{2, 4}, 3, 5, {3, 7}};
    auto val_symbols = set_shape_symbols(value_shape);
    PartialShape attention_mask_shape{{2, 7}, 3, {4, 7}, 5};
    set_shape_symbols(attention_mask_shape);
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, query_shape);
    const auto key = std::make_shared<opset13::Parameter>(element::f64, key_shape);
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, value_shape);
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f64, attention_mask_shape);
    const auto scale = std::make_shared<opset13::Parameter>(element::f64, PartialShape{-1});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 3, {4, 5}, {3, 7}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(symbols[0], symbols[1], symbols[2], val_symbols[3]));
}

TEST(type_prop, scaled_dot_product_attention_mixed_shape_infer_5_inputs_ignore_attention) {
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{{1, 4}, 3, {1, 5}, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f64, PartialShape{{4, 8}, {1, 4}, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{{2, 4}, 3, 5, {3, 7}});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::i64, PartialShape{57, 3, {4, 7}, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f64, PartialShape{});
    auto causal = true;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 3, {1, 5}, {3, 7}}));
}

TEST(type_prop, scaled_dot_product_attention_infer_5_dynamic_attn_partial) {
    const auto dynamic = PartialShape::dynamic();
    const auto query = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto key = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto value = std::make_shared<opset13::Parameter>(element::f32, dynamic);
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, -1, 5, 7});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (dynamic));
}

TEST(type_prop, scaled_dot_product_attention_mixed_shape_infer_4_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{{1, 4}, 4, {2, 5}, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f64, PartialShape{{2, 8}, {1, 4}, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{{2, 4}, 4, 5, {3, 7}});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f64, PartialShape{4, {4, 7}, 5});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{2, 4}, 4, {4, 5}, {3, 7}}));
}

TEST(type_prop, scaled_dot_product_attention_type_infer_5_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{1, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_type_infer_4_inputs) {
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f64, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f64, PartialShape{3, 5});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_type_infer_4_inputs_bool_attention) {
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f64, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::boolean, PartialShape{1, 3, 5});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_broadcast_attention_L) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, 16, 32});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, 48, 32});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, 48, 64});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, 1, 48});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 8, 16, 64}));
}

TEST(type_prop, scaled_dot_product_attention_dyn_broadcast_attention_L) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, {8, -1}, 32});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, -1, 32});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, -1, 64});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 8, 1, -1});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 8, {8, -1}, 64}));
}

TEST(type_prop, scaled_dot_product_unsupported_key_shape) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{3, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{3, 3, 3, 5});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal),
        AssertFailure,
        testing::HasSubstr("Key input shape not compatible with other inputs."));
}
TEST(type_prop, scaled_dot_product_unsupported_value_shape) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{3, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{3, 3, 3, 5});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal),
        AssertFailure,
        testing::HasSubstr("Value input shape not compatible with other inputs."));
}

TEST(type_prop, scaled_dot_product_unsupported_attention_shape) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{3, 3, 3, 5});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal),
        AssertFailure,
        testing::HasSubstr("Attention mask input shape not compatible with other inputs."));
}

TEST(type_prop, scaled_dot_product_unsupported_scale_shape) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op =
            std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal),
        AssertFailure,
        testing::HasSubstr("Scale input must be scalar or have 1 element."));
}

TEST(type_prop, scaled_dot_product_unsupported_dtype) {
    const auto query = std::make_shared<opset13::Parameter>(element::i32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::i32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::i32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::boolean, PartialShape{3, 3, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::i32, PartialShape{});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op =
            std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal),
        AssertFailure,
        testing::HasSubstr("The element type of the input tensor must be a floating-point."));
}

TEST(type_prop, scaled_dot_product_unsupported_value_dtype_mixed) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f64, PartialShape{3, 3, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op =
            std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal),
        AssertFailure,
        testing::HasSubstr("Mixed input types are not supported."));
}

TEST(type_prop, scaled_dot_product_unsuported_attention_type) {
    const auto query = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f64, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::dynamic, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::i32, PartialShape{1, 3, 5});
    auto causal = false;

    OV_EXPECT_THROW(
        auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal),
        AssertFailure,
        testing::HasSubstr("The element type of attention_mask must be either floating-point or boolean."));
}
