// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, TileTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{3}, std::vector<int>{3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({18, 32, 10}));
    // Test Wrong Static Shape
    std::vector<StaticShape> wrong_static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{}},
                             wrong_static_output_shapes = {StaticShape{}};

    ASSERT_THROW(shape_inference(tile.get(), wrong_static_input_shapes, wrong_static_output_shapes), ov::AssertFailure);
}

TEST(StaticShapeInferenceTest, TileFewRepeatsTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = ov::op::v0::Constant::create(element::i64, Shape{2}, {4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({6, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileSmallDataRankTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto param1 = ov::op::v0::Constant::create(element::i64, Shape{3}, {3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{8, 10}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 32, 10}));
}