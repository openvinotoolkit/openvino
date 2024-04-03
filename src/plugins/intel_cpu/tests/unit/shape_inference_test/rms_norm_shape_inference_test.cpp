// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Parameter;

TEST(StaticShapeInferenceTest, RMSNormStaticShapeInferenceTest2ins) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}};
    int32_t axis_val = -1;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes, const_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 8, 6}));
}

TEST(StaticShapeInferenceTest, RMSNormStaticShapeInferenceTest3ins) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{1});
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}, StaticShape{1}};
    int32_t axis_val = -1;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes, const_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 8, 6}));
}
