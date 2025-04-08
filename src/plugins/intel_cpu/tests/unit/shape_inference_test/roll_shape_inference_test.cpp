// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>

#include <array>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class RollV7StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v7::Roll> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(RollV7StaticShapeInferenceTest, axes_as_constant) {
    const auto arg = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto shift = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    const auto axes = Constant::create(element::i64, ov::Shape{2}, {-2, 1});

    const auto op = make_op(arg, shift, axes);

    input_shapes = {StaticShape{3, 5}, StaticShape{2}, StaticShape{2}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], input_shapes[0]);
}

TEST_F(RollV7StaticShapeInferenceTest, axes_in_const_map) {
    const auto arg = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto shift = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());

    const auto op = make_op(arg, shift, axes);
    auto axes_val = std::array<int32_t, 3>{0, 1, -1};

    const auto constant_data =
        std::unordered_map<size_t, ov::Tensor>{{2, {element::i32, ov::Shape{axes_val.size()}, axes_val.data()}}};

    input_shapes = {StaticShape{3, 3, 3}, StaticShape{3}, StaticShape{3}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], input_shapes[0]);
}

TEST_F(RollV7StaticShapeInferenceTest, axes_over_arg_rank) {
    const auto arg = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto shift = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());

    const auto op = make_op(arg, shift, axes);
    auto axes_val = std::array<int32_t, 3>{0, 3, -1};

    const auto constant_data =
        std::unordered_map<size_t, ov::Tensor>{{2, {element::i32, ov::Shape{axes_val.size()}, axes_val.data()}}};

    input_shapes = {StaticShape{3, 3, 3}, StaticShape{3}, StaticShape{3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, constant_data),
                    NodeValidationFailure,
                    HasSubstr("Axis 3 out of the tensor rank range"));
}

TEST_F(RollV7StaticShapeInferenceTest, axes_has_negative_after_normalization) {
    const auto arg = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto shift = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape::dynamic());

    const auto op = make_op(arg, shift, axes);

    auto axes_val = std::array<int64_t, 3>{-4, 2, -1};
    const auto constant_data =
        std::unordered_map<size_t, ov::Tensor>{{2, {element::i64, ov::Shape{axes_val.size()}, axes_val.data()}}};

    input_shapes = {StaticShape{3, 3, 3}, StaticShape{3}, StaticShape{3}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, constant_data),
                    NodeValidationFailure,
                    HasSubstr("Axis -4 out of the tensor rank range"));
}

TEST_F(RollV7StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    auto axes_val = std::array<int64_t, 4>{-4, 2, -1, 1};
    const auto constant_data =
        std::unordered_map<size_t, ov::Tensor>{{2, {element::i64, ov::Shape{axes_val.size()}, axes_val.data()}}};

    input_shapes = {StaticShape{3, 2, 5, 1}, StaticShape{}, StaticShape{4}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);
    EXPECT_EQ(output_shapes[0], input_shapes[0]);
}
