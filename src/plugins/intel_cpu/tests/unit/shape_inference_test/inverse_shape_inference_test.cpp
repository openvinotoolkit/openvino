// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inverse_shape_inference.hpp"

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, InverseStaticShapeInferenceTest2D) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 4});
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Static Shape 2D input
    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 4}};
    auto static_output_shapes = shape_infer(inverse.get(), static_input_shapes, make_tensor_accessor());
    ASSERT_EQ(static_output_shapes[0], StaticShape({4, 4}));
}

TEST(StaticShapeInferenceTest, InverseDynamicShapeInferenceTestAllDimKnown2D) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4, 4});
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Partial Shape 2D input, unknown num_samples
    std::vector<PartialShape> partial_input_shapes = {PartialShape{4, 4}};
    auto partial_output_shapes = shape_infer(inverse.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({4, 4}));
}

TEST(StaticShapeInferenceTest, InverseDynamicShapeInferenceTestAllDimKnown3D) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{10, 4, 4});
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Partial Shape 2D input
    std::vector<PartialShape> partial_input_shapes = {PartialShape{10, 4, 4}};
    auto partial_output_shapes = shape_infer(inverse.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({10, 4, 4}));
}

TEST(StaticShapeInferenceTest, InverseDynamicShapeInferenceTestDynamicDims2D) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Partial Shape 2D input, unknown num_samples and input shape
    std::vector<PartialShape> partial_input_shapes = {PartialShape{-1, -1}};
    auto partial_output_shapes = shape_infer(inverse.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({-1, -1}));
}

TEST(StaticShapeInferenceTest, InverseDynamicShapeInferenceTestStaticBatchDynamicDims2D) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{10, -1, -1});
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Partial Shape 2D input, unknown num_samples and input shape
    std::vector<PartialShape> partial_input_shapes = {PartialShape{10, -1, -1}};
    auto partial_output_shapes = shape_infer(inverse.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({10, -1, -1}));
}

TEST(StaticShapeInferenceTest, InverseDynamicShapeInferenceTestDynamicDims3D) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Partial Shape 2D input, unknown num_samples and input shape
    std::vector<PartialShape> partial_input_shapes = {PartialShape{-1, -1, -1}};
    auto partial_output_shapes = shape_infer(inverse.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({-1, -1, -1}));
}

TEST(StaticShapeInferenceTest, InverseDynamicShapeInferenceTestDynamicRank) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto inverse = std::make_shared<op::v14::Inverse>(input, false);

    // Test Partial Shape dynamic input, unknown num_samples and input shape
    std::vector<PartialShape> partial_input_shapes = {PartialShape::dynamic()};
    auto partial_output_shapes = shape_infer(inverse.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape::dynamic());
}
