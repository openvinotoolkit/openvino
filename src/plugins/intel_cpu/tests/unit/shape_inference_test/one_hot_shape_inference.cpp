// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <one_hot_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, OneHotTest) {
    auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto depth = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto on_value = op::v0::Constant::create(element::u32, Shape{}, {5});
    auto off_value = op::v0::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = -1;
    auto ont_hot = std::make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3}, StaticShape{}, StaticShape{}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(ont_hot.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{3, 2}));
}