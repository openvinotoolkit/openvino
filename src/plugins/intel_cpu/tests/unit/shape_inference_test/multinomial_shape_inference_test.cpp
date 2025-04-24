// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial_shape_inference.hpp"

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, MultinomialStaticShapeInferenceTest2D) {
    auto probs = std::make_shared<op::v0::Parameter>(element::f32, ov::Shape{4, 4});
    auto num_samples = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{1});
    auto multinomial = std::make_shared<op::v13::Multinomial>(probs, num_samples, element::i32, false, false, 0, 0);

    // Test Static Shape 2D input
    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 4}, StaticShape{1}};
    int32_t num_elements_val = 2;
    auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, ov::Shape{1}, &num_elements_val}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(multinomial.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({4, 2}));
}

TEST(StaticShapeInferenceTest, MultinomialDynamicShapeInferenceTestAllDimKnown2D) {
    auto probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3});
    auto num_samples = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    auto multinomial = std::make_shared<op::v13::Multinomial>(probs, num_samples, element::i32, false, false, 0, 0);

    // Test Partial Shape 2D input
    std::vector<PartialShape> partial_input_shapes = {PartialShape{2, 3}, PartialShape{1}};
    int32_t num_elements_val = 2;
    auto const_data = std::unordered_map<size_t, Tensor>{{1, {element::i32, ov::Shape{1}, &num_elements_val}}};
    auto acc = make_tensor_accessor(const_data);
    auto partial_output_shapes = shape_infer(multinomial.get(), partial_input_shapes, acc);
    ASSERT_EQ(partial_output_shapes[0], PartialShape({2, 2}));
}

TEST(StaticShapeInferenceTest, MultinomialDynamicShapeInferenceTestDynamicNumSamples2D) {
    auto probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4, 4});
    auto num_samples = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto multinomial = std::make_shared<op::v13::Multinomial>(probs, num_samples, element::i32, false, false, 0, 0);

    // Test Partial Shape 2D input, unknown num_samples
    std::vector<PartialShape> partial_input_shapes = {PartialShape{4, 4}, PartialShape{-1}};
    auto partial_output_shapes = shape_infer(multinomial.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({4, -1}));
}

TEST(StaticShapeInferenceTest, MultinomialDynamicShapeInferenceTestDynamicProbsDynamicNumSamples2D) {
    auto probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto num_samples = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto multinomial = std::make_shared<op::v13::Multinomial>(probs, num_samples, element::i32, false, false, 0, 0);

    // Test Partial Shape 2D input, unknown num_samples and probs shape
    std::vector<PartialShape> partial_input_shapes = {PartialShape{-1, -1}, PartialShape{-1}};
    auto partial_output_shapes = shape_infer(multinomial.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape({-1, -1}));
}

TEST(StaticShapeInferenceTest, MultinomialDynamicShapeInferenceTestDynamicProbsDynamicNumSamplesDynamicRank) {
    auto probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto num_samples = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto multinomial = std::make_shared<op::v13::Multinomial>(probs, num_samples, element::i32, false, false, 0, 0);

    // Test Partial Shape dynamic input, unknown num_samples and probs shape
    std::vector<PartialShape> partial_input_shapes = {PartialShape::dynamic(), PartialShape{-1}};
    auto partial_output_shapes = shape_infer(multinomial.get(), partial_input_shapes, make_tensor_accessor());
    ASSERT_EQ(partial_output_shapes[0], PartialShape::dynamic());
}
