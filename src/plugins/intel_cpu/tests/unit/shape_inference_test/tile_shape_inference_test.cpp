// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, TileTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{3}, std::vector<int>{3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{3}};
    const auto static_output_shapes = shape_inference(tile.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({18, 32, 10}));
    // Test Wrong Static Shape
    std::vector<StaticShape> wrong_static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{}};

    ASSERT_THROW(shape_inference(tile.get(), wrong_static_input_shapes), ov::AssertFailure);
}

TEST(StaticShapeInferenceTest, TileFewRepeatsTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = ov::op::v0::Constant::create(element::i64, ov::Shape{2}, {4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{2}};
    const auto static_output_shapes = shape_inference(tile.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({6, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileSmallDataRankTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto param1 = ov::op::v0::Constant::create(element::i64, ov::Shape{3}, {3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto static_output_shapes = shape_inference(tile.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileSmallDataRankTestRepeatsInConstMap) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, repeats}}};

    // Test Static Shape
    StaticShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}}, output_shapes = {StaticShape{}};
    output_shapes = shape_inference(tile.get(), input_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), StaticShape({3, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileStaticShapeRepeatsAsConst) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto param1 = ov::op::v0::Constant::create(element::i64, ov::Shape{2}, {4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    auto dims = std::vector<VectorDims>{{6, 8, 10}, {2}};
    auto in_shapes = std::vector<StaticShapeRef>(dims.begin(), dims.end());

    const auto op_infer = make_shape_inference(tile);
    const auto outputs = op_infer->infer(in_shapes, ov::make_tensor_accessor());

    ASSERT_TRUE(outputs);
    EXPECT_EQ(outputs->size(), 1);
    EXPECT_EQ(outputs->front(), StaticShape({6, 32, 10}));
    EXPECT_EQ(*outputs->front(), VectorDims({6, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileNewApiInputsStaticRank) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1, 2};
    const auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{4}, repeats}}};

    auto dims = std::vector<VectorDims>{{8, 10}, {4}};
    auto in_shapes = std::vector<StaticShapeRef>(dims.begin(), dims.end());

    const auto op_infer = make_shape_inference(tile);
    const auto outputs = op_infer->infer(in_shapes, ov::make_tensor_accessor(constant_data));

    ASSERT_TRUE(outputs);
    EXPECT_EQ(outputs->size(), 1);
    EXPECT_EQ(outputs->front(), StaticShape({3, 4, 8, 20}));
    EXPECT_EQ(*outputs->front(), VectorDims({3, 4, 8, 20}));
}
