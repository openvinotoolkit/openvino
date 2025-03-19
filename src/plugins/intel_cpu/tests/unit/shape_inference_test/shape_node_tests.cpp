// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

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
