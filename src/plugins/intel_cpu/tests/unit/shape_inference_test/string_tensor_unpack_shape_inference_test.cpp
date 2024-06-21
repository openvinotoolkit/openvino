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

class StringTensorUnpackStaticTestSuite : public ::testing::TestWithParam<std::tuple<
                                                              std::vector<std::string>,         // input string
                                                              Shape,                            // input shape
                                                              Shape                             // expected output shape
                                                              >> {};

class StringTensorUnpackStaticShapeInferenceTest: public OpStaticShapeInferenceTest<op::v15::StringTensorUnpack> {};

TEST_P(StringTensorUnpackStaticTestSuite, StringTensorUnpackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& input_string = std::get<0>(param);
    const auto& input_shape = std::get<1>(param);
    const auto& expected_string_length = std::get<2>(param);

    const auto data = std::make_shared<Constant>(element::string, input_shape, input_string);
    const auto op = std::make_shared<op::v15::StringTensorUnpack>(data);
    const auto input_shapes = ShapeVector{input_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes.front(), StaticShape(input_shape));
    EXPECT_EQ(output_shapes[1], StaticShape(input_shape));
    EXPECT_EQ(output_shapes[2], StaticShape(expected_string_length));
}

//TEST_F(StringTensorUnpackStaticShapeInferenceTest, kernel_size_and_output_size_from_tensor_accessor) {
//    const auto data = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
//    const auto output_size = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
//    const auto kernel_size = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
//    const auto strides = Strides{2, 2};
//    const auto dilations = Strides{2, 2};
//    const auto pads_begin = Shape{2, 2};
//    const auto pads_end = Shape{2, 2};
//    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
//
//    int64_t output_size_val[] = {32, 32};
//    int64_t kernel_size_val[] = {2, 2};
//    auto const_inputs = std::unordered_map<size_t, Tensor>{{1, {element::i64, Shape{2}, output_size_val}},
//                                                           {2, {element::i64, Shape{2}, kernel_size_val}}};
//
//    const auto input_shapes = ShapeVector{Shape{3, 12, 289}, Shape{2}, Shape{2}};
//    auto shape_infer = make_shape_inference(op);
//    const auto input_shape_refs = make_static_shape_refs(input_shapes);
//    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
//    EXPECT_EQ(output_shapes.size(), 1);
//    EXPECT_EQ(output_shapes.front(), StaticShape({3, 3, 32, 32}));
//}

INSTANTIATE_TEST_SUITE_P(
    StringTensorUnpackStaticShapeInferenceTests,
    StringTensorUnpackStaticTestSuite,
    ::testing::Values(
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO"},
            PartialShape{2},
            PartialShape{13})
    )
);
