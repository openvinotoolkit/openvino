// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"
#include "string_tensor_unpack_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;

static size_t get_character_count(const std::vector<std::string>& vec) {
    size_t count = 0;
    for (const auto& str : vec) {
        count += str.size();
    }
    return count;
}

class StringTensorUnpackStaticTestSuite : public ::testing::TestWithParam<std::tuple<
                                                              std::vector<std::string>,         // input data
                                                              Shape                             // input shape
                                                              >> {};

class StringTensorUnpackStaticShapeInferenceTest: public OpStaticShapeInferenceTest<op::v15::StringTensorUnpack> {};

TEST_F(StringTensorUnpackStaticShapeInferenceTest, data_from_tensor_accessor_1) {
    const auto data = std::make_shared<Parameter>(element::string, ov::PartialShape::dynamic());
    const auto op = make_op(data);
    std::string data_val[] = {"Intel", "OpenVINO"};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::string, Shape{2}, data_val}}};

    const auto input_shapes = ShapeVector{Shape{2}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({2}));
    EXPECT_EQ(output_shapes[1], StaticShape({2}));
    EXPECT_EQ(output_shapes[2], StaticShape({13}));
}

TEST_F(StringTensorUnpackStaticShapeInferenceTest, data_from_tensor_accessor_2) {
    const auto data = std::make_shared<Parameter>(element::string, ov::PartialShape::dynamic());
    const auto op = make_op(data);
    std::string data_val[] = {"Intel Corp", "   ", "Open VINO", "", "Artificial Intelligence"};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::string, Shape{5}, data_val}}};

    const auto input_shapes = ShapeVector{Shape{5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({5}));
    EXPECT_EQ(output_shapes[1], StaticShape({5}));
    EXPECT_EQ(output_shapes[2], StaticShape({45}));
}

TEST_F(StringTensorUnpackStaticShapeInferenceTest, data_from_tensor_accessor_3) {
    const auto data = std::make_shared<Parameter>(element::string, ov::PartialShape::dynamic());
    const auto op = make_op(data);
    std::string data_val[] = {"Intel", "OpenVINO", "AI", "Edge", "Compute", "Vision", "Neural", "Networks"};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::string, Shape{2, 2, 2}, data_val}}};

    const auto input_shapes = ShapeVector{Shape{2, 2, 2}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 2, 2}));
    EXPECT_EQ(output_shapes[1], StaticShape({2, 2, 2}));
    EXPECT_EQ(output_shapes[2], StaticShape({46}));
}

TEST_F(StringTensorUnpackStaticShapeInferenceTest, data_from_tensor_accessor_4) {
    const auto data = std::make_shared<Parameter>(element::string, ov::PartialShape::dynamic());
    const auto op = make_op(data);
    std::string data_val[] = {"In@tel", "Open#VINO", "A$I"};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::string, Shape{1, 3}, data_val}}};

    const auto input_shapes = ShapeVector{Shape{1, 3}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 3}));
    EXPECT_EQ(output_shapes[1], StaticShape({1, 3}));
    EXPECT_EQ(output_shapes[2], StaticShape({18}));
}

TEST_P(StringTensorUnpackStaticTestSuite, StringTensorUnpackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& input_strings = std::get<0>(param);
    const auto& input_shape = std::get<1>(param);

    const auto data = std::make_shared<Constant>(element::string, input_shape, input_strings);
    const auto op = std::make_shared<op::v15::StringTensorUnpack>(data);
    const auto input_shapes = ShapeVector{input_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape(input_shape));
    EXPECT_EQ(output_shapes[1], StaticShape(input_shape));
    EXPECT_EQ(output_shapes[2], StaticShape({get_character_count(input_strings)}));
}

INSTANTIATE_TEST_SUITE_P(
    StringTensorUnpackStaticShapeInferenceTests,
    StringTensorUnpackStaticTestSuite,
    ::testing::Values(
        // single string
        std::make_tuple(
            std::vector<std::string>{"Intel"},
            Shape{1}),
        // multiple strings
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO", "AI"},
            Shape{3}),
        // empty string
        std::make_tuple(
            std::vector<std::string>{""},
            Shape{1}),
        // strings with special characters
        std::make_tuple(
            std::vector<std::string>{"In@tel", "Open#VINO", "A$I"},
            Shape{3}),
        // strings with spaces and an empty string
        std::make_tuple(
            std::vector<std::string>{"Intel Corp", "   ", "Open VINO", "", "Artificial Intelligence"},
            Shape{1, 5}),
        // empty vector
        std::make_tuple(
            std::vector<std::string>{},
            Shape{0}),
        // different shapes
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO", "AI", "Edge"},
            Shape{2, 2}),
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO", "AI", "Edge", "Compute", "Vision"},
            Shape{2, 3}),
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO", "AI", "Edge", "Compute", "Vision", "Neural", "Networks"},
            Shape{2, 2, 2}),
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO", "AI", "Edge"},
            Shape{1, 4}),
        std::make_tuple(
            std::vector<std::string>{"Intel"},
            Shape{1, 1})
    )
);
