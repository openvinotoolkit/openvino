// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/shape_of.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ReshapeTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, -1});

    auto reduce =
            std::make_shared<op::v1::Reshape>(data, pattern, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{2}},
            static_output_shapes = {StaticShape{}};
    shape_inference(reduce.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 150}));
}

TEST(StaticShapeInferenceTest, SqueezeTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{-3, 0});

    auto reduce =
            std::make_shared<op::v0::Squeeze>(data, pattern);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 6, 1, 7, 1}, StaticShape{2}},
            static_output_shapes = {StaticShape{}};
    shape_inference(reduce.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({6, 7, 1}));
}

TEST(StaticShapeInferenceTest, UnsqueezeTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto pattern = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{-3, 0});

    auto reduce =
            std::make_shared<op::v0::Unsqueeze>(data, pattern);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}, StaticShape{2}},
            static_output_shapes = {StaticShape{}};
    shape_inference(reduce.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 3, 4, 1, 5, 6}));
}

TEST(StaticShapeInferenceTest, ShapeOf5DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto shapeof =
            std::make_shared<op::v0::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}},
            static_output_shapes = {StaticShape{}};
    shape_inference(shapeof.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({5}));
}

TEST(StaticShapeInferenceTest, ShapeOf0DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{});

    auto shapeof =
            std::make_shared<op::v3::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{}},
            static_output_shapes = {StaticShape{}};
    shape_inference(shapeof.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({}));
}
