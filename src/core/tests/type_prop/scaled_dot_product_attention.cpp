// Copyright (C) 2018-2023 Intel Corporation
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
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 6}));
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

TEST(type_prop, scaled_dot_product_attention_static_iopored_attention_mask) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 8, 9, 10, 11});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

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
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 1, 3, 5});
    auto causal = false;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_3_inputs_extra_batch) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 3, 4});
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
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 6});
    auto causal = true;

    const auto op = std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
}

TEST(type_prop, scaled_dot_product_attention_static_iopored_attention_mask_extra_batch) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, 7, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{7, 8, 9, 10, 11});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{});
    auto causal = false;

    const auto op =
        std::make_shared<opset13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 7, 3, 6}));
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
