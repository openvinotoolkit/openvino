// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ReshapeTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, -1});

    auto reduce = std::make_shared<op::v1::Reshape>(data, pattern, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{2}};
    const auto static_output_shapes = shape_inference(reduce.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 150}));
}

TEST(StaticShapeInferenceTest, ReshapeEmptyTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 2, 2});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 4});

    auto reduce = std::make_shared<op::v1::Reshape>(data, pattern, false);

    std::vector<StaticShape> static_input_shapes = {StaticShape{0, 2, 2}, StaticShape{2}};
    const auto static_output_shapes = shape_inference(reduce.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({0, 4}));
}

TEST(StaticShapeInferenceTest, ShapeOf5DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto shapeof = std::make_shared<op::v0::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}};
    const auto static_output_shapes = shape_inference(shapeof.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({5}));
}

TEST(StaticShapeInferenceTest, ShapeOf0DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{});

    auto shapeof = std::make_shared<op::v3::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{}};
    const auto static_output_shapes = shape_inference(shapeof.get(), static_input_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({}));
}
