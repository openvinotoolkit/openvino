// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class ScaledDotProductAttentionV13StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v13::ScaledDotProductAttention> {
};

TEST_F(ScaledDotProductAttentionV13StaticShapeInferenceTest, default_ctor) {
    op = make_op();

    input_shapes = ShapeVector{{3, 2, 3, 4}, {2, 5, 4}, {1, 5, 6}, {1, 3, 5}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 2, 3, 6}));
}

TEST_F(ScaledDotProductAttentionV13StaticShapeInferenceTest, dynamic_shapes) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape{-1});
    auto causal = false;
    op = make_op(query, key, value, attention_mask, scale, causal);

    input_shapes = ShapeVector{{2, 3, 4}, {2, 5, 4}, {2, 5, 6}, {1, 3, 5}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 3, 6}));
}

TEST_F(ScaledDotProductAttentionV13StaticShapeInferenceTest, static_shapes) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 3, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 5, 4});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, Shape{1, 3, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, Shape{1});
    auto causal = false;
    op = make_op(query, key, value, attention_mask, scale, causal);

    input_shapes = ShapeVector{{2, 3, 4}, {2, 5, 4}, {2, 5, 6}, {1, 3, 5}, {1}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 3, 6}));
}

TEST_F(ScaledDotProductAttentionV13StaticShapeInferenceTest, mixed_shapes) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape{2, {2, 3}, 4});
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape{-1, {2, 7}, -1});
    const auto value = std::make_shared<opset13::Parameter>(element::f32, Shape{2, 5, 6});
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, {3, 5}, 5});
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, Shape{});
    auto causal = false;
    op = make_op(query, key, value, attention_mask, scale, causal);

    input_shapes = ShapeVector{{2, 3, 4}, {2, 5, 4}, {2, 5, 6}, {1, 3, 5}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 3, 6}));
}

TEST_F(ScaledDotProductAttentionV13StaticShapeInferenceTest, attention_L_broadcast) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    auto causal = false;

    op = make_op(query, key, value, attention_mask, scale, causal);

    input_shapes = ShapeVector{{2, 8, 16, 32}, {2, 8, 24, 32}, {2, 8, 24, 48}, {1, 1, 24}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 8, 16, 48}));
}

TEST_F(ScaledDotProductAttentionV13StaticShapeInferenceTest, attention_S_broadcast) {
    const auto query = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto key = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto value = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto attention_mask = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    const auto scale = std::make_shared<opset13::Parameter>(element::f32, PartialShape::dynamic());
    auto causal = false;

    op = make_op(query, key, value, attention_mask, scale, causal);

    input_shapes = ShapeVector{{2, 8, 16, 32}, {2, 8, 24, 32}, {2, 8, 24, 48}, {1, 16, 1}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 8, 16, 48}));
}
