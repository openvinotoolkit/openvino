// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/rms_norm.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

TEST(StaticShapeInferenceTest, RMSNormStaticShapeInferenceTestDefaultCtor) {
    const auto op = std::make_shared<op::internal::RMSNorm>();
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());

    op->set_arguments(ov::OutputVector{data, axes, scale});

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}, StaticShape{1}};
    int32_t axis_val = -1;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes, const_data);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 3, 8, 6}));
}

TEST(StaticShapeInferenceTest, RMSNormStaticShapeInferenceTest2ins) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}};
    int32_t axis_val = -1;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes, const_data);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 3, 8, 6}));
}

TEST(StaticShapeInferenceTest, RMSNormStaticShapeInferenceTest3ins) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, scale, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}, StaticShape{1}};
    int32_t axis_val = -1;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes, const_data);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 3, 8, 6}));
}

TEST(StaticShapeInferenceTest, RMSNormIncorrectAxisValParam) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}};
    int32_t axis_val = 5;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};

    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Axis 5 out of the tensor rank range [-4, 3]"));
}

TEST(StaticShapeInferenceTest, RMSNormIncorrectAxisValConst) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Constant>(element::i32, Shape{}, 5);
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{}};

    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Axis 5 out of the tensor rank range [-4, 3]"));
}

TEST(StaticShapeInferenceTest, RMSNormIncorrectAxisShapeDim) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{5}};
    int32_t axis_val = 5;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};

    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Number of the axes can't be higher than the rank of the data shape"));
}

TEST(StaticShapeInferenceTest, RMSNormIncorrectAxisShapeRank) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1, 5}};
    int32_t axis_val = 5;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};

    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Axes input must be a scalar or 1D input. Got: {1,5}"));
}

TEST(StaticShapeInferenceTest, RMSNormIncorrectScaleShape) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::internal::RMSNorm>(data, axes, scale, eps);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 8, 6}, StaticShape{1}, StaticShape{6, 1}};
    int32_t axis_val = -1;
    const auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, Shape{1}, &axis_val}}};

    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("Scale input shape must be broadcastable to the shape of the data input"));
}
